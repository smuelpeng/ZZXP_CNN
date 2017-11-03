/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//#ifdef MKL2017_SUPPORTED
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "../../filler.hpp"
#include "../../layer.hpp"
#include "mkl_convolution_layer.hpp"
#include "mkl_service.h"
#include "caffe/profiler.hpp"
static int getMKLBuildDate() {
  static int build = 0;
  if (build == 0) {
    MKLVersion v;
    mkl_get_version(&v);
    build = atoi(v.Build);
  }
  return build;
}

namespace caffe {
MKLConvolutionLayer::MKLConvolutionLayer(
  const LayerParameter& param)
      : ConvolutionLayer(param),
        fwd_bottom_data(new MKLData<real_t>()),
        fwd_top_data(new MKLData<real_t>()),
        fwd_filter_data(new MKLData<real_t>()),
        fwd_bias_data(new MKLData<real_t>()),
        convolutionFwd(NULL)
        {}

void MKLConvolutionLayer::compute_output_shape() {
  ConvolutionLayer::compute_output_shape();
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

void MKLConvolutionLayer::CreateFwdPrimitive() {
  int status;
  size_t g = std::max(this->group_, 1);
  size_t dimension = 4;
  // Free MKL primitives
  dnnDelete<real_t>(convolutionFwd);
  if (this->bias_term_) {
    status = dnnGroupsConvolutionCreateForwardBias<real_t>(
      &convolutionFwd,
      NULL,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  } else {
    status = dnnGroupsConvolutionCreateForward<real_t>(
      &convolutionFwd,
      NULL,
      dnnAlgorithmConvolutionDirect,
      g,
      dimension,
      bdata_sizes,
      tdata_sizes,
      fdata_sizes,
      convolutionStrides,
      inputOffset,
      dnnBorderZeros);
  }

  CHECK_EQ(status, 0)
          << "Failed dnnCreateConvolution<real_t>(dnnForward) with sta_ttus "
          << status << "\n";

  fwd_bottom_data->create_layouts(convolutionFwd, dnnResourceSrc, dimension,
                                  bdata_sizes, bdata_strides);
  fwd_top_data   ->create_layouts(convolutionFwd, dnnResourceDst, dimension,
                                  tdata_sizes, tdata_strides);
  fwd_filter_data->create_layouts(convolutionFwd, dnnResourceFilter,
                                  f_dimension, fdata_sizes, fdata_strides);

  if (this->bias_term_)
    fwd_bias_data->create_layouts(convolutionFwd, dnnResourceBias, 1,
                                  bias_sizes, bias_strides);

}

MKLConvolutionLayer::~MKLConvolutionLayer() {
    dnnDelete<real_t>(convolutionFwd);
}

void MKLConvolutionLayer::Init(
      const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  this->width_ = bottom[0]->width();
  this->height_ = bottom[0]->height();
  this->num_ = bottom[0]->num();

  // TODO: clean up this
  kernel_w_ = this->kernel_shape_.cpu_data()[1];
  kernel_h_ = this->kernel_shape_.cpu_data()[0];
  stride_w_ = this->stride_.cpu_data()[1];
  stride_h_ = this->stride_.cpu_data()[0];
  pad_w_ = this->pad_.cpu_data()[1];
  pad_h_ = this->pad_.cpu_data()[0];

  this->bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;
  size_t kw, kh; /* filter */
  size_t dimension = 4;

  g  = std::max(this->group_, 1);
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_;

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_;

  kw = this->kernel_w_;
  kh = this->kernel_h_;

  this->bdata_sizes[0] = iw;
  this->bdata_sizes[1] = ih;
  this->bdata_sizes[2] = ic;
  this->bdata_sizes[3] = n;

  this->bdata_strides[0] = 1;
  this->bdata_strides[1] = iw;
  this->bdata_strides[2] = iw*ih;
  this->bdata_strides[3] = iw*ih*ic;

  /* starting with MKL 2017 Gold in case of groups filter layout
   * becomes 5D, i.e. groups become a separate dimension */
  size_t g_mkl2017 = g;
  f_dimension = dimension + (g != 1);
  if (getMKLBuildDate() < 20160701) {
      g_mkl2017 = 1;
      f_dimension = dimension;
  }

  this->fdata_sizes[0] = kw;
  this->fdata_sizes[1] = kh;
  this->fdata_sizes[2] = ic/g;
  this->fdata_sizes[3] = oc/g_mkl2017;
  this->fdata_sizes[4] = g_mkl2017;

  this->fdata_strides[0] = 1;
  this->fdata_strides[1] = kw;
  this->fdata_strides[2] = kw*kh;
  this->fdata_strides[3] = kw*kh*ic/g;
  this->fdata_strides[4] = kw*kh*ic/g*oc/g;

  this->bias_sizes[0] = oc;

  this->bias_strides[0] = 1;

  this->tdata_sizes[0] = ow;
  this->tdata_sizes[1] = oh;
  this->tdata_sizes[2] = oc;
  this->tdata_sizes[3] = n;

  this->tdata_strides[0]  = 1;
  this->tdata_strides[1]  = ow;
  this->tdata_strides[2]  = ow*oh;
  this->tdata_strides[3]  = ow*oh*oc;

  this->convolutionStrides[0] = this->stride_w_;
  this->convolutionStrides[1] = this->stride_h_;

  this->inputOffset[0] = -this->pad_w_;
  this->inputOffset[1] = -this->pad_h_;

  // Names are for debugging purposes only.
  fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
  fwd_filter_data ->name = "fwd_filter_data   @ " + this->layer_param_.name();
  fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();

  CreateFwdPrimitive();
}

void MKLConvolutionLayer::LayerSetUp(
      const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  ConvolutionLayer::LayerSetUp(bottom, top);
  Init(bottom, top);
}

void MKLConvolutionLayer::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  this->reshape = (this->width_ == bottom[0]->width() &&
                   this->height_ == bottom[0]->height() &&
                   this->channels_ == bottom[0]->channels() &&
                   this->num_ == bottom[0]->num()) ? false : true;

  BaseConvolutionLayer::Reshape(bottom, top);

  if (this->reshape == true) {
    // when reshape happens, sync weight and bias data/diff to cpu.
    this->blobs_[0]->mutable_cpu_data();
    if (this->bias_term_) {
      this->blobs_[1]->mutable_cpu_data();
    }
    Init(bottom, top);
  }
}

void MKLConvolutionLayer::Forward_cpu(
  const vector<Blob*>& bottom, const vector<Blob*>& top) {
  int status;
  size_t n, g;
  size_t iw, ih, ic;
  size_t ow, oh, oc;

  g  = this->group_;
  n  = this->num_;
  iw = this->width_;
  ih = this->height_;
  ic = this->channels_/g;

  CHECK(bottom[0]->width()    == iw &&
        bottom[0]->height()   == ih &&
        bottom[0]->channels() == ic*g &&
        bottom[0]->num()      == n)
          << "Inclompatible shape of bottom with layer";

  ow = this->width_out_;
  oh = this->height_out_;
  oc = this->num_output_/g;
  CHECK(top[0]->width()    == ow &&
        top[0]->height()   == oh &&
        top[0]->channels() == oc*g &&
        top[0]->num()      == n) << "Inclompatible shape of bottom with layer";


  void *res_convolutionFwd[dnnResourceNumber];
  res_convolutionFwd[dnnResourceSrc] =
    fwd_bottom_data->get_converted_prv(bottom[0], false);
  res_convolutionFwd[dnnResourceFilter] =
    fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);
  if (this->bias_term_) {
    res_convolutionFwd[dnnResourceBias] =
      fwd_bias_data  ->get_converted_prv(this->blobs_[1].get(), true);
  }

  if (fwd_top_data->conversion_needed()) {
    top[0]->set_prv_data_descriptor(fwd_top_data);
    res_convolutionFwd[dnnResourceDst] =
            reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
    res_convolutionFwd[dnnResourceDst] = top[0]->mutable_cpu_data();
  }
  Profiler *profiler = Profiler::Get();
  profiler->ScopeStart("MKlconvolution_dnnExecute");
  status = dnnExecute<real_t>(convolutionFwd, res_convolutionFwd);
  profiler->ScopeEnd();
  CHECK_EQ(status, 0) << "Forward convolution failed with status " << status;
}

#ifdef CPU_ONLY
STUB_GPU(MKLConvolutionLayer);
#else
void MKLConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
  {NOT_IMPLEMENTED;}
#endif
}  // namespace caffe
//#endif  // #ifdef MKL2017_SUPPORTED
