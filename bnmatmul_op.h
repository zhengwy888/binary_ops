#pragma once

#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor_types.h>
#include <tensorflow/core/framework/op_kernel.h>

namespace tensorflow {

template <typename T>
struct MatMulTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned> in_type;
};

void XNORGemmKernelDevice(OpKernelContext *ctx, const float* in0, const float* in1, const int m, const int n, const int k, float* out);
}
