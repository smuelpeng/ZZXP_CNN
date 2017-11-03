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
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "../../filler.hpp"
#include "../../layer.hpp"
#include "mkl_deconvolution_layer.hpp"
#include "mkl_service.h"
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
MKLDeconvolutionLayer::MKLDeconvolutionLayer(
  const LayerParameter& param)
      : DeconvolutionLayer(param),
        fwd_bottom_data(new MKLData<real_t>()),
        fwd_top_data(new MKLData<real_t>()),
        fwd_filter_data(new MKLData<real_t>()),
        fwd_bias_data(new MKLData<real_t>()),
        convolutionBwdData(NULL)
        {}

void MKLDeconvolutionLayer::compute_output_shape() {
  DeconvolutionLayer::compute_output_shape();
  this->height_out_ = this->stride_h_ * (this->height_ - 1)
      + this->kernel_h_ - 2 * this->pad_h_ ;
  this->width_out_ = this->stride_w_ * (this->width_ - 1)
      + this->kernel_w_ - 2 * this->pad_w_ ;
}

MKLDeconvolutionLayer::~MKLDeconvolutionLayer() {
    dnnDelete<real_t>(convolutionBwdData);
}

void MKLDeconvolutionLayer::Init(
      const vector<Blob*>& bottom,
      const vector<Blob*>& top) {

#ifdef _OPENMP
  this->num_of_threads_ = omp_get_max_threads() < bottom[0]->shape(0) ?
                    omp_get_max_threads() : bottom[0]->shape(0);
  if (this->num_of_threads_ < 1) {
     LOG(WARNING) << "DeConv layer: omp_get_max_threads() ="
                  << this->num_of_threads_;
     this->num_of_threads_ = 1;
  }
#endif


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
  int status;
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

  size_t bdata_sizes[4] = {iw, ih, ic, n};
  size_t bdata_strides[4] = {1, iw, iw*ih, iw*ih*ic};

  /* starting with MKL 2017 Gold in case of groups filter layout
   * becomes 5D, i.e. groups become a separate dimension */
  size_t g_mkl2017 = g;
  size_t f_dimension = dimension + (g != 1);
  if (getMKLBuildDate() < 20160701) {
      g_mkl2017 = 1;
      f_dimension = dimension;
  }

  size_t fdata_sizes[5] = {kw, kh, oc/g, ic/g_mkl2017, g_mkl2017};
  size_t fdata_strides[5]  = {1, kw, kw*kh, kw*kh*oc/g, kw*kh*ic/g*oc/g};

  size_t bias_sizes[1] = {oc};
  size_t bias_strides[1] = {1};

  size_t tdata_sizes[4] = {ow, oh, oc, n};
  size_t tdata_strides[4]  = {1, ow, ow*oh, ow*oh*oc};

  size_t convolutionStrides[2] = {this->stride_w_, this->stride_h_};
  int    inputOffset[2] = {-this->pad_w_, -this->pad_h_};

  // Names are for debugging purposes only.
  fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
  fwd_filter_data ->name = "fwd_filter_data   @ " + this->layer_param_.name();
  fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();

/*
 * Forward setup, implemented by convolutionBwdData
 */
  dnnDelete<real_t>(convolutionBwdData);
  status = dnnGroupsConvolutionCreateBackwardData<real_t>(
    &convolutionBwdData,
    NULL,
    dnnAlgorithmConvolutionDirect,
    g,
    dimension,
    tdata_sizes,
    bdata_sizes,
    fdata_sizes,
    convolutionStrides,
    inputOffset,
    dnnBorderZeros);
  CHECK_EQ(status, 0)
          << "Failed dnnConvolutionCreateBackwardData with status "
          << status << "\n";
  fwd_bottom_data->create_layouts(convolutionBwdData, dnnResourceDiffDst, dimension,
                                  bdata_sizes, bdata_strides);
  fwd_top_data   ->create_layouts(convolutionBwdData, dnnResourceDiffSrc, dimension,
                                  tdata_sizes, tdata_strides);
  fwd_filter_data->create_layouts(convolutionBwdData, dnnResourceFilter,
                                  f_dimension, fdata_sizes, fdata_strides);

}

void MKLDeconvolutionLayer::LayerSetUp(
      const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  DeconvolutionLayer::LayerSetUp(bottom, top);

  Init(bottom, top);
}

void MKLDeconvolutionLayer::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  bool reinitialize = (this->width_ == bottom[0]->width() &&
                       this->height_ == bottom[0]->height() &&
                       this->channels_ == bottom[0]->channels() &&
                       this->num_ == bottom[0]->num()) ? false : true;

  BaseConvolutionLayer::Reshape(bottom, top);

  if (reinitialize == true) {
    this->blobs_[0]->mutable_cpu_data();
    if (this->bias_term_) {
      this->blobs_[1]->mutable_cpu_data();
    }
    Init(bottom, top);
  }
}

void MKLDeconvolutionLayer::Forward_cpu(
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


  void *res_convolutionBwdData[dnnResourceNumber];

  res_convolutionBwdData[dnnResourceDiffDst] =
      fwd_bottom_data->get_converted_prv(bottom[0], false);
  // Currently this conversion adds padding to weights.
  // We don't want that to be stored in the weights prv_ptr_
  res_convolutionBwdData[dnnResourceFilter]  =
      fwd_filter_data->get_converted_prv(this->blobs_[0].get(), true);

  if (fwd_top_data->conversion_needed()) {
      top[0]->set_prv_data_descriptor(fwd_top_data);
      res_convolutionBwdData[dnnResourceDiffSrc] =
          reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
      res_convolutionBwdData[dnnResourceDiffSrc] =
          top[0]->mutable_cpu_data();
  }

  status = dnnExecute<real_t>(convolutionBwdData, res_convolutionBwdData);
  CHECK_EQ(status, 0) << "Forward deconvolution failed with status " << status;

  if (this->bias_term_) {
      const real_t* bias = this->blobs_[1]->cpu_data();
      real_t* top_data = top[0]->mutable_cpu_data();

#ifdef _OPENMP
#   pragma omp parallel for num_threads(this->num_of_threads_)
#endif
      for (int n = 0; n < this->num_; ++n) {
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDeconvolutionLayer);
#else
void MKLDeconvolutionLayer::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top)
  {NOT_IMPLEMENTED;}
#endif
}  // namespace caffe
