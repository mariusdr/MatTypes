#ifndef LIBCUMANIP_TYPES_LINALG_H
#define LIBCUMANIP_TYPES_LINALG_H

#include "matrix.hpp"

namespace cumanip
{
namespace mt 
{

template <size_t N>
__host__ __device__ inline bool invert(const Matrix<N, N>& mat, Matrix<N, N>& inverted);

template <size_t N>
__host__ __device__ inline Matrix<N, N> pivot_matrix(const Matrix<N, N>& inp);

template <size_t N> 
__host__ __device__ inline void LU_decomposition(const Matrix<N, N>& inp, Matrix<N, N>& p, Matrix<N, N>& l, Matrix<N, N>& u);

template <size_t N> 
__host__ __device__ inline float signum(const Matrix<N, N>& inp);

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
bool is_singular(const Matrix<Rows, Cols>& mat, float eps=0.000001);

template <size_t N> 
__host__ __device__ inline float determinant(const Matrix<N, N>& inp, float eps=0.000001);

template <size_t N>
__host__ __device__ inline float approx_eigenvalue(const Matrix<N, N>& m, size_t iterations, Vector<N>& ev);

template <size_t N> 
__host__ __device__ inline void min_max_eigenvalues(const Matrix<N, N>& m, size_t iterations, float& min_lambda, float& max_lambda);


//==================================================================================================================//
// Impl                                                                                                             //
//==================================================================================================================//

// helper function for matrix inversion
template <size_t Rows, size_t Cols>
__host__ __device__ inline 
bool swap_lines_2darray(float a[Rows][Cols], size_t r1, size_t r2)
{
    if (r1 >= Rows || r2 >= Rows)
    {
        return false;
    }

    for (size_t j = 0; j < Cols; ++j)
    {
        float tmp = a[r1][j];
        a[r1][j] = a[r2][j];
        a[r2][j] = tmp;
    }
    return true;
}

// from http://www.virtual-maxim.de/matrix-invertieren-in-c-plus-plus/
template <size_t N>
__host__ __device__ inline 
bool invert(const Matrix<N, N>& mat, Matrix<N, N>& inverted)
{
    float A[N][2 * N];
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
            A[i][j] = mat.data[i * N + j];
        for (size_t j = N; j < 2 * N; ++j)
            A[i][j] = (i == j - N) ? 1.0 : 0.0;
    }

    for (size_t k = 0; k < N - 1; ++k)
    {
        if (A[k][k] == 0.0)
        {
            for (size_t i = k + 1; i < N; ++i)
            {
                if (A[i][k] != 0.0)
                {
                    swap_lines_2darray<N, 2 * N>(A, k, i);
                    break;
                }
                else if (i == N - 1)
                    return false;
            }
        }

        for (size_t i = k + 1; i < N; ++i)
        {
            float p = A[i][k] / A[k][k];
            for (size_t j = k; j < 2 * N; ++j)
                A[i][j] -= A[k][j] * p;
        }
    }

    float det = 1.0;
    for (size_t k = 0; k < N; ++k)
        det *= A[k][k];

    if (det == 0.0) 
        return false;

    for (size_t k = N - 1; k > 0; --k)
    {
        for (int i = k - 1; i >= 0; --i)
        {
            float p = A[i][k] / A[k][k];
            for (size_t j = k; j < 2 * N; ++j)
                A[i][j] -= A[k][j] * p;
        }
    }

    for (size_t i = 0; i < N; ++i)
    {
        const float f = A[i][i];
        for (size_t j = N; j < 2 * N; ++j)
            inverted.data[i * N + (j - N)] = A[i][j] / f;
    }

    return true;
}

template <size_t N>
__host__ __device__ inline
Matrix<N, N> pivot_matrix(const Matrix<N, N>& inp)
{
    Matrix<N, N> p = identity<N, N>();

    for (size_t j = 0; j < N; ++j)
    {
        size_t i = j;
        for (size_t it = j; it < N; ++it)
        {
            const float& x = inp.data[i * N + j];
            const float& xt = inp.data[it * N + j];
            if (fabs(xt) > fabs(x))
            {
                i = it;
            }
        }

        if (j != i)
        {
            auto tmp = p.get_row(j);
            p.set_row(j, p.get_row(i));
            p.set_row(i, tmp);
        }
    }

    return p;
}

// from https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/
template <size_t N> 
__host__ __device__ inline
void LU_decomposition(const Matrix<N, N>& inp, Matrix<N, N>& p, Matrix<N, N>& l, Matrix<N, N>& u)
{
    p = pivot_matrix<N>(inp);
    l = m_zeros<N, N>();
    u = m_zeros<N, N>();

    Matrix<N, N> pa = p * inp;

    for (size_t j = 0; j < N; ++j)
    {
        l.data[j * N + j] = 1.f;

        for (size_t i = 0; i < j + 1; ++i)
        {
            float sum = 0.f;
            for (size_t k = 0; k < i; ++k)
            {
                sum += u.data[k * N + j] * l.data[i * N + k];
            }

            u.data[i * N + j] = pa.data[i * N + j] - sum;
        }

        for (size_t i = j; i < N; ++i)
        {
            float sum = 0.f;
            for (size_t k = 0; k < j; ++k)
            {
                sum += u.data[k * N + j] * l.data[i * N + k];
            }
            
            l.data[i * N + j] = (pa.data[i * N + j] - sum) / u.data[j * N + j];
        }
    }
}

// this fails if the input is not a permuation matrix 
template <size_t N> 
__host__ __device__ inline
float signum(const Matrix<N, N>& inp)
{
    int perm[N];
    
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (inp.data[i * N + j] > 0.f)
            {
                perm[i] = j;
            }
        }
    }

    int sum = 0;
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = i + 1; j < N; ++j)
        {
            if (perm[i] > perm[j])
            {
                sum += 1;
            }
        }
    }

    return powf(-1, sum);
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
bool is_singular(const Matrix<Rows, Cols>& mat, float eps)
{
    bool singular = false;
    for (size_t i = 0; i < Rows; ++i)
    {
        mt::Vector<Cols> r = mat.get_row(i);
        Vector<Cols> zero(0.f);
        singular |= r.approx_equal(zero);
    }
    return singular;
}

template <size_t N> 
__host__ __device__ inline
float determinant(const Matrix<N, N>& inp, float eps)
{
    Matrix<N, N> p, l, u;
    bool sng = is_singular(inp);
    LU_decomposition(inp, p, l, u);

    const float dp = signum(p);
    
    float dl = 1.f;
    float du = 1.f;

    for (size_t i = 0; i < N; ++i)
    {
        dl = l.data[i * N + i] * dl;
        du = u.data[i * N + i] * du; 
    }

    if (fabs(dl) < eps)
    {
        dl = 0.f;
    }

    if (fabs(du) < eps)
    {
        du = 0.f;
    }

    if (sng) 
    {
        du = 0.f;
        dl = 0.f;
    }

    return dp * dl * du;
}

template <size_t N>
__host__ __device__ inline 
float approx_eigenvalue(const Matrix<N, N>& m, size_t iterations, Vector<N>& ev)
{
    for (size_t i = 0; i < N; ++i)
    {
        ev.data[i] = float(i) / float(N);
    }

    for (size_t c = 0; c < iterations; ++c)
    {
        ev = m * ev;
        ev = ev.normalized();
    }

    // rayleigh coefficient
    Vector<N> lambda_ev = m * ev;
    float lambda = ev.dot(lambda_ev) / ev.dot(ev);
    return lambda;
}

template <size_t N> 
__host__ __device__ inline
void min_max_eigenvalues(const Matrix<N, N>& m, size_t iterations, float& min_lambda, float& max_lambda)
{
    Vector<N> tmp;
    max_lambda = approx_eigenvalue(m, iterations, tmp);
    Matrix<N, N> mi;
    invert<N>(m, mi);
    min_lambda = 1.f / approx_eigenvalue(mi, iterations, tmp);
}




} // namespace mt
} // namespace cumanip
#endif