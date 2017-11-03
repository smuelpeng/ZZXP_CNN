#ifndef CAFFE_MKL2017_DECONVOLUTION_LAYERS_HPP_
#define CAFFE_MKL2017_DECONVOLUTION_LAYERS_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "../../common.hpp"
#include "../base_conv_layer.hpp"
#include "../deconv_layer.hpp"

#include "../../proto/caffe.pb.h"

#include "../../mkl_memory.hpp"
#include "../../mkl_dnn_cppwrapper.h"

namespace caffe {
class MKLDeconvolutionLayer : public DeconvolutionLayer {
 public:
  explicit MKLDeconvolutionLayer(const LayerParameter& param);

  virtual ~MKLDeconvolutionLayer();

  virtual inline const char* type() const { return "MklDeconvolution"; }

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

 private:
  /* Fwd step */
  shared_ptr<MKLData<real_t> > fwd_bottom_data, fwd_top_data, fwd_filter_data,
                                 fwd_bias_data;
  dnnPrimitive_t convolutionBwdData;

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
};
}
#endif