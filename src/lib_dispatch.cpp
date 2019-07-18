#include "lib_dispatch.hpp"
#include <iostream>
#include <type_traits>

#ifdef ASGARD_BUILD_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

struct device_handler
{
  cublasHandle_t handle;

  device_handler()
  {
    auto success = cublasCreate(&handle);
    assert(success == 0);

    success = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    assert(success == 0);
  }
  ~device_handler() { cublasDestroy(handle); }
};
static device_handler device;

inline cublasOperation_t cublas_trans(char trans)
{
  if (trans == 'N' || trans == 'n')
  {
    return CUBLAS_OP_N;
  }
  else
  {
    return CUBLAS_OP_T;
  }
}

#endif

auto const ignore = [](auto ignored) { (void)ignored; };

namespace lib_dispatch
{
template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy, resource const res)
{
  assert(n);
  assert(x);
  assert(incx);
  assert(y);
  assert(incy);
  assert(*incx >= 0);
  assert(*incy >= 0);
  assert(*n >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA

    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // function instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDcopy(device.handle, *n, x, *incx, y, *incy);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasScopy(device.handle, *n, x, *incx, y, *incy);
      assert(success == 0);
    }
    return;
#endif
  }

  // host execution
  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(n, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    scopy_(n, x, incx, y, incy);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      y[i * (*incy)] = x[i * (*incx)];
    }
  }
}

template<typename P>
P dot(int *n, P *x, int *incx, P *y, int *incy, resource const res)
{
  assert(n);
  assert(x);
  assert(incx);
  assert(y);
  assert(incy);
  assert(*incx >= 0);
  assert(*incy >= 0);
  assert(*n >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    P result;
    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDdot(device.handle, *n, x, *incx, y, *incy, &result);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSdot(device.handle, *n, x, *incx, y, *incy, &result);
      assert(success == 0);
    }
    return result;

#endif
  }
  if constexpr (std::is_same<P, double>::value)
  {
    return ddot_(n, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    return sdot_(n, x, incx, y, incy);
  }
  else
  {
    P ans = 0.0;
    for (int i = 0; i < *n; ++i)
    {
      ans += x[i * (*incx)] * y[i * (*incy)];
    }
    return ans;
  }
}

template<typename P>
void axpy(int *n, P *alpha, P *x, int *incx, P *y, int *incy,
          resource const res)
{
  assert(n);
  assert(alpha);
  assert(x);
  assert(incx);
  assert(y);
  assert(incy);
  assert(*incx >= 0);
  assert(*incy >= 0);
  assert(*n >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDaxpy(device.handle, *n, alpha, x, *incx, y, *incy);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSaxpy(device.handle, *n, alpha, x, *incx, y, *incy);
      assert(success == 0);
    }
    return;
#endif
  }

  if constexpr (std::is_same<P, double>::value)
  {
    daxpy_(n, alpha, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    saxpy_(n, alpha, x, incx, y, incy);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      y[i * (*incy)] = y[i * (*incy)] + x[i * (*incx)] * (*alpha);
    }
  }
}

template<typename P>
void scal(int *n, P *alpha, P *x, int *incx, resource const res)
{
  assert(n);
  assert(alpha);
  assert(x);
  assert(incx);
  assert(*n >= 0);
  assert(*incx >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDscal(device.handle, *n, alpha, x, *incx);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSscal(device.handle, *n, alpha, x, *incx);
      assert(success == 0);
    }
    return;
#endif
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dscal_(n, alpha, x, incx);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sscal_(n, alpha, x, incx);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      x[i * (*incx)] *= *alpha;
    }
  }
}

//
// Simple helpers for non-float types
//
template<typename P>
static void
basic_gemm(P const *A, bool const trans_A, int const lda, P *B, bool trans_B,
           int const ldb, P *C, int const ldc, int const m, int const k,
           int const n, P const alpha, P const beta)
{
  assert(m > 0);
  assert(k > 0);
  assert(n > 0);
  assert(lda > 0); // FIXME Tyler says these could be more thorough
  assert(ldb > 0);
  assert(ldc > 0);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      P result = 0.0;
      for (auto z = 0; z < k; ++z)
      {
        int const A_loc = trans_A ? i * lda + z : z * lda + i;
        int const B_loc = trans_B ? z * ldb + j : j * ldb + z;
        result += A[A_loc] * B[B_loc];
      }
      C[j * ldc + i] = C[j * ldc + i] * beta + alpha * result;
    }
  }
}

template<typename P>
static void basic_gemv(P const *A, bool const trans_A, int const lda,
                       P const *x, int const incx, P *y, int const incy,
                       int const m, int const n, P const alpha, P const beta)
{
  assert(m > 0);
  assert(n > 0);
  assert(lda > 0);
  assert(incx > 0);
  assert(incy > 0);

  for (auto i = 0; i < m; ++i)
  {
    P result = 0.0;
    for (auto j = 0; j < n; ++j)
    {
      int const A_loc = trans_A ? i * lda + j : j * lda + i;
      result += A[A_loc] * x[j * incx];
    }
    y[i * incy] = y[i * incy] * beta + alpha * result;
  }
}

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy, resource const res)
{
  assert(trans);
  assert(m);
  assert(n);
  assert(alpha);
  assert(A);
  assert(lda);
  assert(x);
  assert(incx);
  assert(beta);
  assert(y);
  assert(incy);
  assert(*m >= 0);
  assert(*n >= 0);
  assert(*lda >= 0);
  assert(*incx >= 0);
  assert(*incy >= 0);
  assert(*trans == 't' || *trans == 'n');

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDgemv(device.handle, cublas_trans(*trans), *m, *n, alpha, A,
                      *lda, x, *incx, beta, y, *incy);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSgemv(device.handle, cublas_trans(*trans), *m, *n, alpha, A,
                      *lda, x, *incx, beta, y, *incy);
      assert(success == 0);
    }
    return;
#endif
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  }
  else
  {
    bool const trans_A = (*trans == 't') ? true : false;
    int const rows_A   = trans_A ? *n : *m;
    int const cols_A   = trans_A ? *m : *n;
    basic_gemv(A, trans_A, *lda, x, *incx, y, *incy, rows_A, cols_A, *alpha,
               *beta);
  }
}

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          resource const res)
{
  assert(transa);
  assert(transb);
  assert(m);
  assert(n);
  assert(k);
  assert(alpha);
  assert(A);
  assert(lda);
  assert(B);
  assert(ldb);
  assert(beta);
  assert(C);
  assert(ldc);
  assert(*m >= 0);
  assert(*n >= 0);
  assert(*k >= 0);
  assert(*transa == 't' || *transa == 'n');
  assert(*transb == 't' || *transb == 'n');

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemm(device.handle, cublas_trans(*transa),
                                       cublas_trans(*transb), *m, *n, *k, alpha,
                                       A, *lda, B, *ldb, beta, C, *ldc);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemm(device.handle, cublas_trans(*transa),
                                       cublas_trans(*transb), *m, *n, *k, alpha,
                                       A, *lda, B, *ldb, beta, C, *ldc);
      assert(success == 0);
    }
    return;
#endif
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else
  {
    bool const trans_A = (*transa == 't') ? true : false;
    bool const trans_B = (*transb == 't') ? true : false;
    basic_gemm(A, trans_A, *lda, B, trans_B, *ldb, C, *ldc, *m, *k, *n, *alpha,
               *beta);
  }
}

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const res)
{
  assert(m);
  assert(n);
  assert(A);
  assert(lda);
  assert(ipiv);
  assert(info);
  assert(*lda >= 0);
  assert(*m >= 0);
  assert(*n >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA

    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);
    assert(*m == *n);
    ignore(m);

    P **A_d;
    auto stat = cudaMalloc((void **)&A_d, sizeof(P *));
    assert(stat == 0);
    stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    assert(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDgetrfBatched(device.handle, *n, A_d, *lda, ipiv, info, 1);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSgetrfBatched(device.handle, *n, A_d, *lda, ipiv, info, 1);
      assert(success == 0);
    }
    return;
#endif
  }

  if constexpr (std::is_same<P, double>::value)
  {
    dgetrf_(m, n, A, lda, ipiv, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetrf_(m, n, A, lda, ipiv, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrf not implemented for non-floating types" << '\n';
    assert(false);
  }
}

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const res)
{
  assert(n);
  assert(A);
  assert(lda);
  assert(ipiv);
  assert(work);
  assert(lwork);
  assert(info);
  assert(*lda >= 0);
  assert(*n >= 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA

    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    assert(*lwork == (*n) * (*n));
    ignore(lwork);

    P **A_d;
    P **work_d;
    auto stat = cudaMalloc((void **)&A_d, sizeof(P *));
    assert(stat == 0);
    stat = cudaMalloc((void **)&work_d, sizeof(P *));
    assert(stat == 0);

    stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    assert(stat == 0);
    stat = cudaMemcpy(work_d, &work, sizeof(P *), cudaMemcpyHostToDevice);
    assert(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgetriBatched(device.handle, *n, A_d, *lda,
                                               nullptr, work_d, *n, info, 1);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgetriBatched(device.handle, *n, A_d, *lda,
                                               nullptr, work_d, *n, info, 1);
      assert(success == 0);
    }
    return;
#endif
  }
  if constexpr (std::is_same<P, double>::value)
  {
    dgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getri not implemented for non-floating types" << '\n';
    assert(false);
  }
}

template<typename P>
void batched_gemm(P **const &a, int *lda, char const *transa, P **const &b,
                  int *ldb, char const *transb, P **const &c, int *ldc, int *m,
                  int *n, int *k, P *alpha, P *beta, int *num_batch,
                  resource const res)
{
  assert(transa);
  assert(transb);
  assert(m);
  assert(n);
  assert(k);
  assert(alpha);
  assert(a);
  assert(lda);
  assert(b);
  assert(ldb);
  assert(beta);
  assert(c);
  assert(ldc);
  assert(*m >= 0);
  assert(*n >= 0);
  assert(*k >= 0);
  assert(*transa == 't' || *transa == 'n');
  assert(*transb == 't' || *transb == 'n');
  assert(*num_batch > 0);

  if (res == resource::device)
  { // device execution (fallback to host)

#ifdef ASGARD_BUILD_CUDA
    // no non-fp blas on device
    assert(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemmBatched(
          device.handle, cublas_trans(*transa), cublas_trans(*transb), *m, *n,
          *k, alpha, a, *lda, b, *ldb, beta, c, *ldc, *num_batch);
      assert(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemmBatched(
          device.handle, cublas_trans(*transa), cublas_trans(*transb), *m, *n,
          *k, alpha, a, *lda, b, *ldb, beta, c, *ldc, *num_batch);
      assert(success == 0);
    }
    return;
#endif
  }

  for (int i = 0; i < *num_batch; ++i)
  {
    gemm(transa, transb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc,
         resource::host);
  }
}

template<typename P>
void batched_gemv(P **a, int *lda, char const *transb, P **b, P **c, int *m,
                  int *n, P *alpha, P *beta, int *num_batch, resource const res)
{}

template void
copy(int *n, float *x, int *incx, float *y, int *incy, resource const res);
template void
copy(int *n, double *x, int *incx, double *y, int *incy, resource const res);
template void
copy(int *n, int *x, int *incx, int *y, int *incy, resource const res);

template float
dot(int *n, float *x, int *incx, float *y, int *incy, resource const res);
template double
dot(int *n, double *x, int *incx, double *y, int *incy, resource const res);
template int
dot(int *n, int *x, int *incx, int *y, int *incy, resource const res);

template void axpy(int *n, float *alpha, float *x, int *incx, float *y,
                   int *incy, resource const res);
template void axpy(int *n, double *alpha, double *x, int *incx, double *y,
                   int *incy, resource const res);
template void axpy(int *n, int *alpha, int *x, int *incx, int *y, int *incy,
                   resource const res);

template void
scal(int *n, float *alpha, float *x, int *incx, resource const res);
template void
scal(int *n, double *alpha, double *x, int *incx, resource const res);
template void scal(int *n, int *alpha, int *x, int *incx, resource const res);

template void gemv(char const *trans, int *m, int *n, float *alpha, float *A,
                   int *lda, float *x, int *incx, float *beta, float *y,
                   int *incy, resource const res);
template void gemv(char const *trans, int *m, int *n, double *alpha, double *A,
                   int *lda, double *x, int *incx, double *beta, double *y,
                   int *incy, resource const res);
template void gemv(char const *trans, int *m, int *n, int *alpha, int *A,
                   int *lda, int *x, int *incx, int *beta, int *y, int *incy,
                   resource const res);

template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, float *alpha, float *A, int *lda, float *B, int *ldb,
                   float *beta, float *C, int *ldc, resource const res);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, double *alpha, double *A, int *lda, double *B,
                   int *ldb, double *beta, double *C, int *ldc,
                   resource const res);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, int *alpha, int *A, int *lda, int *B, int *ldb,
                   int *beta, int *C, int *ldc, resource const res);

template void getrf(int *m, int *n, float *A, int *lda, int *ipiv, int *info,
                    resource const res);
template void getrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info,
                    resource const res);

template void getri(int *n, float *A, int *lda, int *ipiv, float *work,
                    int *lwork, int *info, resource const res);
template void getri(int *n, double *A, int *lda, int *ipiv, double *work,
                    int *lwork, int *info, resource const res);

template void batched_gemm(float **const &a, int *lda, char const *transa,
                           float **const &b, int *ldb, char const *transb,
                           float **const &c, int *ldc, int *m, int *n, int *k,
                           float *alpha, float *beta, int *num_batch,
                           resource const res);

template void batched_gemm(double **const &a, int *lda, char const *transa,
                           double **const &b, int *ldb, char const *transb,
                           double **const &c, int *ldc, int *m, int *n, int *k,
                           double *alpha, double *beta, int *num_batch,
                           resource const res);

template void batched_gemv(float **a, int *lda, char const *transa, float **b,
                           float **c, int *m, int *n, float *alpha, float *beta,
                           int *num_batch, resource res);

template void batched_gemv(double **a, int *lda, char const *transa, double **b,
                           double **c, int *m, int *n, double *alpha,
                           double *beta, int *num_batch, resource res);

} // namespace lib_dispatch
