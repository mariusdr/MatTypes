#ifndef LIBCUMANIP_TYPES_MATRIX_H
#define LIBCUMANIP_TYPES_MATRIX_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>


namespace cumanip
{
namespace mt 
{

template <size_t Rows, size_t Cols>
struct Matrix;

template <size_t Len> 
struct Vector;

using Vector3f = Vector<3>;
using Vector4f = Vector<4>;
using Vector6f = Vector<6>;

using State = Vector<6>;
using Point = Vector<6>;
using Point3 = Vector<3>;

using Matrix3f = Matrix<3, 3>;
using Matrix4f = Matrix<4, 4>;


template <size_t Rows, size_t Cols>
struct Matrix
{
    __host__ __device__ Matrix() {};
    __host__ __device__ explicit Matrix(float val);
    __host__ __device__ explicit Matrix(float* val);
    __host__ __device__ Matrix(const Matrix<Rows, Cols>& rhs);

    __host__ __device__ size_t rows() const { return Rows; }
    __host__ __device__ size_t cols() const { return Cols; }
    __host__ __device__ size_t size() const { return Rows * Cols; };

    __host__ __device__ Matrix<Rows, Cols>& operator=(const Matrix<Rows, Cols>& rhs);

    __host__ __device__ void set_row(size_t i, const Vector<Cols>& row);
    __host__ __device__ void set_col(size_t j, const Vector<Rows>& col);

    __host__ __device__ Vector<Cols> get_row(size_t i) const;
    __host__ __device__ Vector<Rows> get_col(size_t j) const;

    __host__ __device__ Matrix<Cols, Rows> transpose() const;

    __host__ __device__ float trace() const;

    __host__ __device__ bool approx_equal(const Matrix<Rows, Cols>& rhs, float eps=0.0001);

    __host__ __device__ float sum() const;
    __host__ __device__ float avg() const;
    __host__ __device__ float max() const;
    __host__ __device__ float min() const;

    __host__ __device__ float max(size_t& i, size_t& j) const;
    __host__ __device__ float min(size_t& i, size_t& j) const;
    
    __host__ __device__ int nz_count(float eps=0.00000001) const;

    __host__ __device__ Matrix<Rows, Cols> replace(float x, float y, float eps=0.00000001) const;
    
    float data[Rows * Cols];
};

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> permutate_rows(const Matrix<Rows, Cols>& mat, int indices[Rows]);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> min_elems(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> max_elems(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> operator*(float lhs, const Matrix<Rows, Cols>& rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> operator*(const Matrix<Rows, Cols>& lhs, float rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> operator+(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> operator-(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs);

template <size_t N, size_t M, size_t L>
__host__ __device__ inline Matrix<N, L> operator*(const Matrix<N, M>& lhs, const Matrix<M, L> &rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Vector<Rows> operator*(const Matrix<Rows, Cols>& lhs, const Vector<Cols>& rhs);

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> m_zeros();

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> m_ones();

template <size_t Rows, size_t Cols>
__host__ __device__ inline Matrix<Rows, Cols> identity();

__host__ __device__ inline Matrix3f mat3f_rows(Vector3f r1, Vector3f r2, Vector3f r3);
__host__ __device__ inline Matrix3f mat3f_cols(Vector3f c1, Vector3f c2, Vector3f c3);
__host__ __device__ inline Matrix4f mat4f_rows(Vector4f r1, Vector4f r2, Vector4f r3, Vector4f r4);
__host__ __device__ inline Matrix4f mat4f_cols(Vector4f c1, Vector4f c2, Vector4f c3, Vector4f c4);
__host__ __device__ inline Matrix3f mat3f(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22);
__host__ __device__ inline Matrix<3, 4> mat3x4_rows(Vector4f r1, Vector4f r2, Vector4f r3);
__host__ __device__ inline Matrix<4, 3> mat4x3_rows(Vector3f r1, Vector3f r2, Vector3f r3, Vector3f r4);

template <size_t Rows, size_t Cols>
__host__ __device__ inline void printMat(const Matrix<Rows, Cols>& mat, const char* fmt = "%f ");

template <size_t Rows, size_t Cols>
__host__ inline std::ostream& operator<<(std::ostream& os, const Matrix<Rows, Cols>& mat);




//==================================================================================================================//
// Impl                                                                                                             //
//==================================================================================================================//

template <size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Rows, Cols>::Matrix(const Matrix<Rows, Cols>& rhs)
{
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        data[i] = rhs.data[i];
    }
}


template <size_t Rows, size_t Cols>
__host__ __device__ 
bool Matrix<Rows, Cols>::approx_equal(const Matrix<Rows, Cols>& rhs, float eps)
{
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        if (fabs(data[i] - rhs.data[i]) > eps) 
        {
            return false;
        }
    }
    return true;
}


template <size_t Rows, size_t Cols>
__host__ __device__ inline 
void printMat(const Matrix<Rows, Cols>& mat, const char* fmt)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        for (size_t j = 0; j < Cols; ++j)
        {
            printf(fmt, mat.data[i * Cols + j]);
        }
        printf("\n");
    }
}

template <size_t Rows, size_t Cols>
__host__ inline 
std::ostream& operator<<(std::ostream& os, const Matrix<Rows, Cols>& mat)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        for (size_t j = 0; j < Cols; ++j)
        {
            os << mat.data[i * Cols + j] << " ";
        }
        os << "\n";
    }
    return os;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Rows, Cols>& Matrix<Rows, Cols>::operator=(const Matrix<Rows, Cols>& rhs)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        for (size_t j = 0; j < Cols; ++j)
        {
            data[i * Cols + j] = rhs.data[i * Cols + j];
        }
    }
    return *this;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Cols, Rows> Matrix<Rows, Cols>::transpose() const
{
    Matrix<Cols, Rows> tps;
    
    for (size_t i = 0; i < Rows; ++i)
    {
        Vector<Cols> row = get_row(i);
        tps.set_col(i, row);
    }
    return tps;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Rows, Cols>::Matrix(float val)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        for (size_t j = 0; j < Cols; ++j)
        {
            data[i * Cols + j] = val;
        }
    }
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
void Matrix<Rows, Cols>::set_row(size_t i, const Vector<Cols>& row)
{
    for (size_t j = 0; j < Cols; ++j)
    {
        data[i * Cols + j] = row.data[j];
    }
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
void Matrix<Rows, Cols>::set_col(size_t j, const Vector<Rows>& col)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        data[i * Cols + j] = col.data[i];
    }
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Vector<Cols> Matrix<Rows, Cols>::get_row(size_t i) const 
{
    Vector<Cols> v;
    for (size_t j = 0; j < Cols; ++j)
    {
        v.data[j] = data[i * Cols + j];
    }
    return v;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Vector<Rows> Matrix<Rows, Cols>::get_col(size_t j) const 
{
    Vector<Rows> v;
    for (size_t i = 0; i < Rows; ++i)
    {
        v.data[i] = data[i * Cols + j];
    }
    return v;
}

__host__ __device__ inline 
Matrix3f mat3f_rows(Vector3f r1, Vector3f r2, Vector3f r3)
{
    Matrix3f m;
    m.set_row(0, r1);
    m.set_row(1, r2);
    m.set_row(2, r3);
    return m;
}

__host__ __device__ inline 
Matrix3f mat3f_cols(Vector3f c1, Vector3f c2, Vector3f c3)
{
    Matrix3f m;
    m.set_col(0, c1);
    m.set_col(1, c2);
    m.set_col(2, c3);
    return m;
}

__host__ __device__ inline 
Matrix4f mat4f_rows(Vector4f r1, Vector4f r2, Vector4f r3, Vector4f r4)
{
    Matrix4f m;
    m.set_row(0, r1);
    m.set_row(1, r2);
    m.set_row(2, r3);
    m.set_row(3, r4);
    return m;

}

__host__ __device__ inline 
Matrix4f mat4f_cols(Vector4f c1, Vector4f c2, Vector4f c3, Vector4f c4)
{
    Matrix4f m;
    m.set_col(0, c1);
    m.set_col(1, c2);
    m.set_col(2, c3);
    m.set_col(3, c4);
    return m;
}

__host__ __device__ inline 
Matrix3f mat3f(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
{
    return mat3f_rows(vec3f(a00, a01, a02), vec3f(a10, a11, a12), vec3f(a20, a21, a22));
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> m_zeros()
{
    return Matrix<Rows, Cols>(0.f);
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> m_ones()
{
    return Matrix<Rows, Cols>(1.f);
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> identity()
{
    Matrix<Rows, Cols> m = m_zeros<Rows, Cols>();
    size_t n = Rows < Cols ? Rows : Cols;
    for (size_t i = 0; i < n; ++i)
    {
        m.data[i * Cols + i] = 1.f;
    }
    return m;
}

__host__ __device__ inline 
Matrix<4, 3> mat4x3_rows(Vector3f r1, Vector3f r2, Vector3f r3, Vector3f r4)
{
    Matrix<4, 3> mat;
    mat.set_row(0, r1);
    mat.set_row(1, r2);
    mat.set_row(2, r3);
    mat.set_row(3, r4);
    return mat;
}

__host__ __device__ inline 
Matrix<3, 4> mat3x4_rows(Vector4f r1, Vector4f r2, Vector4f r3)
{
    Matrix<3, 4> mat;
    mat.set_row(0, r1);
    mat.set_row(1, r2);
    mat.set_row(2, r3);
    return mat;
}

template<size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Rows, Cols>::Matrix(float* val)
{
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        data[i] = val[i];
    }
}

template <size_t I, size_t K, size_t J>
__host__ __device__ inline 
Matrix<I, J> operator*(const Matrix<I, K>& lhs, const Matrix<K, J> &rhs)
{
    Matrix<I, J> res;

    for (size_t i = 0; i < J; ++i)
    {
        for (size_t j = 0; j < J; ++j)
        {
            float s = 0.f;
            for (size_t k = 0; k < K; ++k)
            {
                const float l = lhs.data[i * K + k];
                const float r = rhs.data[k * J + j];
                s += l * r;
            }
            res.data[i * J + j] = s;
        }
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Vector<Rows> operator*(const Matrix<Rows, Cols>& lhs, const Vector<Cols>& rhs)
{
    Vector<Rows> res;
    for (size_t i = 0; i < Rows; ++i)
    {
        res.data[i] = lhs.get_row(i).dot(rhs);
    }
    return res;
}


__host__ __device__ inline
void coeffs_mat3f(const Matrix3f& mat, float& a11, float& a12, float& a13, float& a21, float& a22, float& a23, float& a31, float& a32, float& a33)
{
    coeffs_vec3f(mat.get_row(0), a11, a12, a13);
    coeffs_vec3f(mat.get_row(1), a21, a22, a23);
    coeffs_vec3f(mat.get_row(3), a31, a32, a33);
}


template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::trace() const
{
    size_t n = Rows < Cols ? Rows : Cols;
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i)
    {
        sum += data[i * Cols + i];
    }
    return sum;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> operator*(float lhs, const Matrix<Rows, Cols>& rhs)
{
    Matrix<Rows, Cols> res;
    size_t n = Rows * Cols;
    for (size_t i = 0; i < n; ++i)
    {
        res.data[i] = lhs * rhs.data[i];
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> operator*(const Matrix<Rows, Cols>& lhs, float rhs)
{
    Matrix<Rows, Cols> res;
    size_t n = Rows * Cols;
    for (size_t i = 0; i < n; ++i)
    {
        res.data[i] = rhs * lhs.data[i];
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> operator+(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs)
{
    Matrix<Rows, Cols> res;
    size_t n = Rows * Cols;
    for (size_t i = 0; i < n; ++i)
    {
        res.data[i] = lhs.data[i] + rhs.data[i];
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> operator-(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs)
{
    Matrix<Rows, Cols> res;
    size_t n = Rows * Cols;
    for (size_t i = 0; i < n; ++i)
    {
        res.data[i] = lhs.data[i] - rhs.data[i];
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::sum() const
{
    float sum = 0.f;
    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            sum += data[i * Cols + j];
        }
    }
    return sum;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::avg() const 
{
    return sum() / float(size());
}

__host__ __device__ inline
float min2(const float x, const float y)
{
    return (x < y) ? x : y;
}

__host__ __device__ inline
float max2(const float x, const float y)
{
    return (x > y) ? x : y;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::max() const 
{
    int m = data[0];
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        m = max2(m, data[i]);
    } 
    return m;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::min() const 
{
    float m = data[0];
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        m = min2(m, data[i]);
    }
    return m;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
int Matrix<Rows, Cols>::nz_count(float eps) const 
{
    int cnt = 0; 
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        if (abs(data[i]) > eps)
        {
            cnt++;
        }
    }
    return cnt;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
Matrix<Rows, Cols> Matrix<Rows, Cols>::replace(float x, float y, float eps) const
{
    Matrix<Rows, Cols> out = *this;
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        if (abs(out.data[i] - x) < eps)
        {
            out.data[i] = y;
        }
    }
    return out;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> min_elems(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs)
{
    Matrix<Rows, Cols> out;
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        out.data[i] = min2(lhs.data[i], rhs.data[i]);
    }
    return out;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> max_elems(const Matrix<Rows, Cols>& lhs, const Matrix<Rows, Cols>& rhs)
{
    Matrix<Rows, Cols> out;
    for (size_t i = 0; i < Rows * Cols; ++i)
    {
        out.data[i] = max2(lhs.data[i], rhs.data[i]);
    }
    return out;
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
Matrix<Rows, Cols> permutate_rows(const Matrix<Rows, Cols>& mat, int indices[Rows])
{
    mt::Matrix<Rows, Cols> res = mat;
    for (int i = 0; i < Rows; ++i)
    {
        res.set_row(indices[i], mat.get_row(i));
    }
    return res;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows,Cols>::max(size_t& i, size_t& j) const 
{
    float maxv = data[0];
    i = 0;
    j = 0;
    for (size_t di = 0; di < Rows; ++di)
    {
        for (size_t dj = 0; dj < Cols; ++dj)
        {
            if (maxv < data[di * Cols + dj])
            {
                maxv = data[di * Cols + dj];
                i = di;
                j = dj;
            }
        }
    }
    return maxv;
}

template <size_t Rows, size_t Cols>
__host__ __device__ 
float Matrix<Rows, Cols>::min(size_t& i, size_t& j) const 
{
    float minv = data[0];
    i = 0;
    j = 0;
    for (size_t di = 0; di < Rows; ++di)
    {
        for (size_t dj = 0; dj < Cols; ++dj)
        {
            if (minv > data[di * Cols + dj])
            {
                minv = data[di * Cols + dj];
                i = di;
                j = dj;
            }
        }
    }
    return minv;
}



} // namespace mt
} // namespace cumanip
#endif