#define EIGEN_USE_THREADS
/* 
 * performance notes:
 * popcnt
 * xnor 1.227158, native 0.114228, dumb 0.000003, reduce 0.432185
 * __builtin_popcnt
 * xnor 1.220534, native 0.109216, dumb 0.000002, reduce 0.424627
 * uint64_t
 * xnor 0.751118, native 0.110391, dumb 0.000003, reduce 0.433113
 * 4 unrolled
 * xnor 0.685397, native 0.109669, dumb 0.000005, reduce 0.444948
 * _mm_popcnt_u64
 * xnor 0.693196, native 0.115468, dumb 0.000005, reduce 0.448353
 * gpu:
 * xnor 0.030549, native 0.101626, dumb 0.000006, reduce 0.23896
 */

#include <stdint.h>
#include <chrono>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "bnmatmul_op.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("DumbMatmul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {half, float, double, int32, complex64, complex128}")
    .Doc(R"doc(
Multiply the matrix "a" by the matrix "b".
The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).
*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.
transpose_a: If true, "a" is transposed before multiplication.
transpose_b: If true, "b" is transposed before multiplication.
)doc");

REGISTER_OP("ReduceMatmul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {half, float, double, int32}")
    .Doc("TODO");
REGISTER_OP("BnMatmul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {half, float, double, int32}")
    .Doc("TODO");


template <typename T>
class DumbMatmulOp : public OpKernel {
 public:
  explicit DumbMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensor
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    //auto input = input_tensor.flat<T>();


    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    // XXX
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;
    OP_REQUIRES(ctx,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    // Create an output tensor
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    auto matA = a.matrix<T>();
    auto matB = b.matrix<T>();
    auto matOut = out->matrix<T>();

    //auto output_flat = out->template flat<T>();

    // Set all the elements of the output tensor to 0
    for ( int r = 0; r < out->dim_size(0); r++ ) 
    {
        for ( int c = 0; c < out->dim_size(1); c++ )
        {
            matOut(r,c) = 0;
            for ( int n = 0; n < a.dim_size(dim_pair[0].first); n++ )
            {
                matOut(r,c) += matA(r, n) * matB(n, c);
            }
        }
    }
  }
};
// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the Op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("DumbMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    DumbMatmulOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("DumbMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    DumbMatmulOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("DumbMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    DumbMatmulOp<double>);

template <typename T>
class ReduceMatmulOp : public OpKernel {
 public:
  explicit ReduceMatmulOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensor
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    //auto input = input_tensor.flat<T>();


    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    // XXX
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;
    OP_REQUIRES(ctx,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    // Create an output tensor
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    auto matA = a.matrix<T>();
    auto matB = b.matrix<T>();
    auto matOut = out->matrix<T>();


    // Set all the elements of the output tensor to 0
    // still not at fast as the native implementation
    matOut.device(ctx->eigen_device<CPUDevice>()) = matA.contract(matB, dim_pair);
  }
private:
  bool transpose_a_;
  bool transpose_b_;
};
// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the Op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ReduceMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ReduceMatmulOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("ReduceMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ReduceMatmulOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ReduceMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    ReduceMatmulOp<double>);

#define INTWIDTH 64
template <typename T>
class BnMatmulOp : public OpKernel {
 public:
  typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MaskMatrix;

  explicit BnMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensor
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);


    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, transpose_a_ == false, errors::InvalidArgument("transpose not supported yet"));
    OP_REQUIRES(ctx, transpose_b_ == false, errors::InvalidArgument("transpose not supported yet"));

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;
    OP_REQUIRES(ctx,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    // Create an output tensor
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    auto matA = a.matrix<T>();
    auto matB = b.matrix<T>();
    auto matOut = out->matrix<T>();

    //auto output_flat = out->template flat<T>();
    auto d = ctx->eigen_device<CPUDevice>();
    concatenate_and_compute(d, matA, matB, matOut);

  }
private:
  bool transpose_a_;
  bool transpose_b_;

  void concatenate_row(
          typename MatMulTypes<T>::in_type array, 
          MaskMatrix &out)
  {
      int colSize = int((array.dimension(1)+INTWIDTH-1 )/ INTWIDTH); 
      out.resize(array.dimension(0), colSize);
      for ( int r = 0; r < array.dimension(0); ++ r )
      {
          for ( int c=0; c< colSize; ++ c)
          {
              uint64_t rvalue=0;
              uint64_t sign;
              for ( int i=0; i< INTWIDTH; ++i ) {
                  int colIdx = c*INTWIDTH + i;
                  if ( colIdx > array.dimension(1)-1 ) {
                      break;
                  }
                  sign = (array(r, colIdx)>=0);
                  rvalue = rvalue | (sign <<i);
              }
              out(r,c) = rvalue;
          }
      }
  }
  void concatenate_row(
          typename MatMulTypes<T>::in_type array, 
          Tensor &out)
  {
      int colSize = int((array.dimension(1)+INTWIDTH-1 )/ INTWIDTH); 
      TensorShape b_shape(
              {array.dimension(0),colSize});
      out.set_shape(b_shape);
      auto out_ = out.matrix<uint64_t>();
      for ( int r = 0; r < array.dimension(0); ++ r )
      {
          for ( int c=0; c< colSize; ++ c)
          {
              uint64_t rvalue=0;
              uint64_t sign;
              for ( int i=0; i< INTWIDTH; ++i ) {
                  int colIdx = c*INTWIDTH + i;
                  if ( colIdx > array.dimension(1)-1 ) {
                      break;
                  }
                  sign = (array(r, colIdx)>=0);
                  rvalue = rvalue | (sign <<i);
              }
              out_(r,c) = rvalue;
          }
      }
  }
  void concatenate_col(
          typename MatMulTypes<T>::in_type array, 
          MaskMatrix &out)
  {
      int rowSize = int((array.dimension(0)+INTWIDTH-1)/ INTWIDTH); 
      out.resize(array.dimension(1),rowSize );

      for ( int c=0; c< array.dimension(1); ++ c)
      {
          for ( int r = 0; r < rowSize; ++ r )
          {
              uint64_t rvalue=0;
              uint64_t sign;
              for ( int i=0; i< INTWIDTH; ++i ) {
                  int rowIdx = r*INTWIDTH + i;
                  if ( rowIdx > array.dimension(0)-1 ) {
                      break;
                  }
                  sign = (array(rowIdx, c )>=0);
                  rvalue = rvalue | (sign <<i);
              }
              out(c, r) = rvalue;
          }
      }
  }
  void concatenate_col(
          typename MatMulTypes<T>::in_type array, 
          Tensor &out)
  {
      int rowSize = int((array.dimension(0)+INTWIDTH-1)/ INTWIDTH); 
      TensorShape b_shape(
              {array.dimension(1),rowSize});
      out.set_shape(b_shape);
      auto out_ = out.matrix<uint64_t>();

      for ( int c=0; c< array.dimension(1); ++ c)
      {
          for ( int r = 0; r < rowSize; ++ r )
          {
              uint64_t rvalue=0;
              uint64_t sign;
              for ( int i=0; i< INTWIDTH; ++i ) {
                  int rowIdx = r*INTWIDTH + i;
                  if ( rowIdx > array.dimension(0)-1 ) {
                      break;
                  }
                  sign = (array(rowIdx, c )>=0);
                  rvalue = rvalue | (sign <<i);
              }
              out_(c, r) = rvalue;
          }
      }
  }
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::milliseconds ms;
  typedef std::chrono::duration<float> fsec;

  void concatenate_and_compute(
          const CPUDevice &d,
          typename MatMulTypes<T>::in_type a,
          typename MatMulTypes<T>::in_type b,
          typename MatMulTypes<T>::out_type out)
  {
      MaskMatrix a_;
      MaskMatrix b_;
      auto t0 = Time::now();
      concatenate_row(a, a_);
      concatenate_col(b, b_);
      auto t1 = Time::now();
      ms d1 = std::chrono::duration_cast<ms>(t1-t0);

      // major time consumer
       //version 1
      int loopsize = int(a_.cols() /4) * 4 ;
      for (int ar=0; ar < a_.rows(); ar++)
      {
          for (int br=0; br< b_.rows(); br++) {
              unsigned int Cvalue = 0;
              for (int c=0; c< loopsize; c += 4) 
              {
                  Cvalue +=__builtin_popcountll(a_(ar, c) ^ b_(br,c));
                  Cvalue +=__builtin_popcountll(a_(ar, c+1) ^ b_(br,c+1));
                  Cvalue +=__builtin_popcountll(a_(ar, c+2) ^ b_(br,c+2));
                  Cvalue +=__builtin_popcountll(a_(ar, c+3) ^ b_(br,c+3));
                  //unsigned int value =popcnt(a_(ar, c) ^ b_(br,c));
                  //unsigned int value =__builtin_popcount(a_(ar, c) ^ b_(br,c));
                  //unsigned int value =__builtin_popcountll(a_(ar, c) ^ b_(br,c));
                  //Cvalue += value;
              }
              for ( int c=loopsize; c< a_.cols(); c++ )
              {
                  Cvalue +=__builtin_popcountll(a_(ar, c) ^ b_(br,c));
              }     
              out(ar, br) = - ( 2*(float)Cvalue - a.dimension(1) );
          }
      }
      auto t2 = Time::now();
      ms d2 = std::chrono::duration_cast<ms>(t2-t1);

  }
};
// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the Op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("BnMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    BnMatmulOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("BnMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    BnMatmulOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("BnMatmul")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    BnMatmulOp<double>);

// GPU implementation
template <typename T>
class BnMatmulGPUOp : public OpKernel {
 public:
  //typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MaskMatrix;
  //typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;

  explicit BnMatmulGPUOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Grab the input tensor
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);


    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    // TODO: support transpose
    OP_REQUIRES(ctx, transpose_a_ == false, errors::InvalidArgument("transpose not supported yet"));
    OP_REQUIRES(ctx, transpose_b_ == false, errors::InvalidArgument("transpose not supported yet"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;
    OP_REQUIRES(ctx,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-compatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    // Create an output tensor
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    auto matA = a.flat<T>().data();
    auto matB = b.flat<T>().data();
    auto matOut = out->flat<T>().data();

    XNORGemmKernelDevice(ctx, matA, matB, a.dim_size(a_dim_remaining), a.dim_size(dim_pair[0].first), b.dim_size(b_dim_remaining),
            matOut);

  }
private:
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_KERNEL_BUILDER(
    Name("BnMatmul")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    BnMatmulGPUOp<float>);

} // namespace tensorflow
