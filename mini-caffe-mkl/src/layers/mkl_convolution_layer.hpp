#ifndef CAFFE_MKL2017_CONVOLUTION_LAYERS_HPP_
#define CAFFE_MKL2017_CONVOLUTION_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "../../common.hpp"
#include "../base_conv_layer.hpp"
#include "../conv_layer.hpp"

#include "../../proto/caffe.pb.h"

#include "../../mkl_memory.hpp"
#include "../../mkl_dnn_cppwrapper.h"

namespace caffe {
class MKLConvolutionLayer : public ConvolutionLayer {
 public:
  explicit MKLConvolutionLayer(const LayerParameter& param);

  virtual ~MKLConvolutionLayer();

  virtual inline const char* type() const { return "MklConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  // Customized methods
  void Init(const vector<Blob*>& bottom,
            const vector<Blob*>& top);

  virtual void LayerSetUp(const vector<Blob*>& bottom,
                          const vector<Blob*>& top);
  virtual void compute_output_shape();

  void Reshape(const vector<Blob*>& bottom,
          const vector<Blob*>& top);

  void CreateFwdPrimitive();
 
 private:
  /* Fwd step */
  shared_ptr<MKLData > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                 fwd_bias_data;
  dnnPrimitive_t convolutionFwd;
  // TODO: temp. compatibility vs. older cafe
  size_t width_,
         height_,
         width_out_,
         height_out_,
         kernel_w_,
         kernel_h_,
         stride_w_,
         stride_h_;
  int    pad_w_,
         pad_h_;

  bool bprop_unpack_called;

  // for reshape
  bool reshape;
  
  size_t bdata_sizes[4];
  size_t bdata_strides[4];

  size_t f_dimension;
  size_t fdata_sizes[5];
  size_t fdata_strides[5];

  size_t bias_sizes[1];
  size_t bias_strides[1];

  size_t tdata_sizes[4];
  size_t tdata_strides[4];

  size_t convolutionStrides[2];
  int    inputOffset[2];

};

}  // namespace caffe
#endif