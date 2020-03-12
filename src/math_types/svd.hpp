#ifndef LIBCUMANIP_TYPES_SVD_HPP
#define LIBCUMANIP_TYPES_SVD_HPP

#include "matrix.hpp"
#include "vector.hpp"
#include "utils.hpp"

namespace cumanip 
{
namespace mt 
{


template<size_t Rows, size_t Cols>
class SVD 
{
public:
    static constexpr size_t NumSVals = Rows < Cols ? Rows : Cols;

    __host__ __device__ explicit SVD(const mt::Matrix<Rows, Cols>& inp);
    __host__ __device__ SVD(const mt::Matrix<Rows, Cols>& inp, size_t iter);

    __host__ __device__ void solve();

    __host__ __device__ 
    mt::Matrix<Rows, Cols> get_inp() const 
    {
        return inp;
    }

    __host__ __device__ 
    mt::Matrix<Rows, Cols> get_u() const
    {
        return u_mat;
    }

    __host__ __device__ 
    mt::Matrix<Rows, Cols> get_s() const
    {
        return s_mat;
    }

    __host__ __device__ 
    mt::Matrix<Cols, Cols> get_v() const
    {
        return v_mat;
    }

    __host__ __device__ 
    mt::Vector<NumSVals> get_singular_values() const 
    {
        mt::Vector<NumSVals> sv(0.f);
        for (size_t i = 0; i < NumSVals; ++i)
        {
            sv.at(i) = s_mat.at(i, i);
        }

        sort_vector_inp(sv);

        return sv;
    }

    __host__ __device__ 
    void set_num_iterations(size_t iter) 
    {
        n_iter = iter;
    }

    __host__ __device__ 
    size_t num_iterations() const 
    {
        return n_iter;
    }

private:
    const mt::Matrix<Rows, Cols> inp;
    mt::Matrix<Rows, Cols> u_mat;
    mt::Matrix<Rows, Cols> s_mat;
    mt::Matrix<Cols, Cols> v_mat; 

    size_t n_iter;
};

template <size_t Rows, size_t Cols> 
__host__ __device__ 
SVD<Rows, Cols>::SVD(const mt::Matrix<Rows, Cols>& inp): SVD(inp, 1000)
{}

template <size_t Rows, size_t Cols> 
__host__ __device__ 
SVD<Rows, Cols>::SVD(const mt::Matrix<Rows, Cols>& inp, size_t iter): 
inp(inp), u_mat(mt::identity<Rows, Cols>()), s_mat(mt::Matrix<Rows, Cols>(0.f)), v_mat(mt::identity<Cols, Cols>()), n_iter(iter)
{}

__host__ __device__
bool isclose(float x, float y, float eps=1e-6) 
{ 
    return fabs(x - y) <= eps * fabs(x + y); 
}

__host__ __device__
void sym_jacobi_coeffs(float x_ii, float x_ij, float x_jj, float* c, float* s) {
    if (!isclose(x_ij, 0)) {
        float tau, t, out_c;
        tau = (x_jj - x_ii) / (2 * x_ij);
        if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1 + tau * tau));
        } else {
            t = -1.0 / (-tau + sqrt(1 + tau * tau));
        }
        out_c = 1.0 / sqrt(1 + t * t);
        *c = out_c;
        *s = t * out_c;
    } else {
        *c = 1.0;
        *s = 0.0;
    }
}

// from https://github.com/ktrianta/jacobi-svd-evd/blob/master/src/svd/one-sided/svd.cpp
template <size_t Rows, size_t Cols> 
__host__ __device__
void SVD<Rows, Cols>::solve()
{
    const size_t m = Rows;
    const size_t n = Cols;
    const size_t n_singular_vals = NumSVals;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            u_mat.at(i, j) = inp.at(i, j);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            v_mat.at(i, j) = 0.f;
            if (i == j) {
                v_mat.at(i, j) = 1.f;
            }
        }
    }

    for (size_t iter = 0; iter < n_iter; ++iter) 
    {
        for (size_t i = 0; i < n - 1; ++i) 
        {
            for (size_t j = i + 1; j < n; ++j) 
            {
                float dot_ii = 0, dot_jj = 0, dot_ij = 0;
                for (size_t k = 0; k < m; ++k) 
                {
                    dot_ii += u_mat.at(k, i) * u_mat.at(k, i);
                    dot_ij += u_mat.at(k, i) * u_mat.at(k, j);
                    dot_jj += u_mat.at(k, j) * u_mat.at(k, j);
                }

                float cosine, sine;
                sym_jacobi_coeffs(dot_ii, dot_ij, dot_jj, &cosine, &sine);

                for (size_t k = 0; k < m; ++k) 
                {
                    float left = cosine * u_mat.at(k, i) - sine * u_mat.at(k, j);
                    float right = sine * u_mat.at(k, i) + cosine * u_mat.at(k, j);
                    u_mat.at(k, i) = left;
                    u_mat.at(k, j) = right;
                }
                for (size_t k = 0; k < n; ++k) 
                {
                    float left = cosine * v_mat.at(k, i) - sine * v_mat.at(k, j);
                    float right = sine * v_mat.at(k, i) + cosine * v_mat.at(k, j);
                    v_mat.at(k, i) = left;
                    v_mat.at(k, j) = right;
                }
            }
        }
    }
    
    float s[NumSVals];

    for (size_t i = 0; i < n; ++i) 
    {
        float sigma = 0.0;
        for (size_t k = 0; k < m; ++k) 
        {
            sigma += u_mat.at(k, i) * u_mat.at(k, i);
        }
        
        sigma = sqrt(sigma);

        if (i < n_singular_vals) 
        {
            s[i] = sigma;
        }

        for (size_t k = 0; k < m; ++k) 
        {
            u_mat.at(k, i) /= sigma;
        }
    }
    
    for (size_t i = 0; i < NumSVals; ++i)
    {
        s_mat.at(i, i) = s[i];
    }
}




} // namespace mt
} // namespace cumanip
#endif