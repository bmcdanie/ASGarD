#pragma once
#include "lib_dispatch.hpp"
#include "tensors.hpp"
#include <climits>

namespace fm
{
// axpy - y = a*x
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &
axpy(fk::vector<P, omem> const &x, fk::vector<P, mem> &y, P const alpha = 1.0)
{
  assert(x.size() == y.size());
  assert(x.size() <= INT_MAX);
  int n    = x.size();
  int one  = 1;
  P alpha_ = alpha;
  lib_dispatch::axpy(&n, &alpha_, x.data(), &one, y.data(), &one);
  return y;
}

// copy(x,y) - copy vector x into y
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &copy(fk::vector<P, omem> const &x, fk::vector<P, mem> &y)
{
  assert(x.size() == y.size());
  assert(x.size() <= INT_MAX);
  int n   = x.size();
  int one = 1;
  lib_dispatch::copy(&n, x.data(), &one, y.data(), &one);
  return y;
}

// scal - scale a vector
template<typename P, mem_type mem>
fk::vector<P, mem> &scal(P const alpha, fk::vector<P, mem> &x)
{
  int one = 1;
  assert(x.size() <= INT_MAX);
  int n    = x.size();
  P alpha_ = alpha;
  lib_dispatch::scal(&n, &alpha_, x.data(), &one);
  return x;
}

// gemv - matrix vector multiplication
// extended to work for matrices whose number of rows is larger than is
// representable with a 32bit int
template<typename P, mem_type amem, mem_type xmem, mem_type ymem>
fk::vector<P, ymem> &
gemv(fk::matrix<P, amem> const &A, fk::vector<P, xmem> const &x,
     fk::vector<P, ymem> &y, bool const trans_A = false, P const alpha = 1.0,
     P const beta = 0.0)
{
  int64_t const cols_A = trans_A ? A.nrows() : A.ncols();
  int64_t const rows_A = trans_A ? A.ncols() : A.nrows();
  assert(rows_A == y.size());
  assert(cols_A == x.size());

  assert(cols_A <= INT_MAX);
  if (rows_A > INT_MAX)
  {
    int const num_iters =
        static_cast<int>(std::ceil(static_cast<double>(A.nrows()) / INT_MAX));
    for (int64_t i = 0; i < num_iters - 1; ++i)
    {
      fk::matrix<P, mem_type::view> A_partial(
          A, i * INT_MAX, (i + 1) * INT_MAX - 1, 0, A.ncols() - 1);
      fk::vector<P, mem_type::view> y_partial(y, i * INT_MAX,
                                              (i + 1) * INT_MAX - 1);
      gemv(A_partial, x, y_partial, trans_A, alpha, beta);
    }
    int64_t const left_over    = A.nrows() % INT_MAX;
    int64_t const already_done = (num_iters - 1) * INT_MAX;
    fk::matrix<P, mem_type::view> A_partial(
        A, already_done, already_done + left_over - 1, 0, A.ncols() - 1);
    fk::vector<P, mem_type::view> y_partial(y, already_done,
                                            already_done + left_over - 1);
    gemv(A_partial, x, y_partial, trans_A, alpha, beta);
  }

  int lda           = A.stride();
  int one           = 1;
  P alpha_          = alpha;
  P beta_           = beta;
  char const transa = trans_A ? 't' : 'n';
  int m             = A.nrows();
  int n             = A.ncols();

  lib_dispatch::gemv(&transa, &m, &n, &alpha_, A.data(), &lda, x.data(), &one,
                     &beta_, y.data(), &one);

  return y;
}

// gemm - matrix matrix multiplication
template<typename P, mem_type amem, mem_type bmem, mem_type cmem>
fk::matrix<P, cmem> &
gemm(fk::matrix<P, amem> const &A, fk::matrix<P, bmem> const &B,
     fk::matrix<P, cmem> &C, bool const trans_A = false,
     bool const trans_B = false, P const alpha = 1.0, P const beta = 0.0)
{
  assert(A.ncols() < INT_MAX);
  assert(A.nrows() < INT_MAX);

  assert(B.ncols() < INT_MAX);
  assert(B.nrows() < INT_MAX);

  int const rows_A = trans_A ? A.ncols() : A.nrows();
  int const cols_A = trans_A ? A.nrows() : A.ncols();

  int const rows_B = trans_B ? B.ncols() : B.nrows();
  int const cols_B = trans_B ? B.nrows() : B.ncols();

  assert(C.nrows() == rows_A);
  assert(C.ncols() == cols_B);
  assert(cols_A == rows_B);

  int lda           = A.stride();
  int ldb           = B.stride();
  int ldc           = C.stride();
  P alpha_          = alpha;
  P beta_           = beta;
  char const transa = trans_A ? 't' : 'n';
  char const transb = trans_B ? 't' : 'n';
  int m             = rows_A;
  int n             = cols_B;
  int k             = rows_B;

  lib_dispatch::gemm(&transa, &transb, &m, &n, &k, &alpha_, A.data(), &lda,
                     B.data(), &ldb, &beta_, C.data(), &ldc);

  return C;
}
} // namespace fm
