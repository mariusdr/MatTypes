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
#include <initializer_list>

#ifndef CHECK_BOUNDS
#define CHECK_BOUNDS
#endif

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

    __host__ Matrix(std::initializer_list<float> l);

    __host__ __device__ float& at(size_t i, size_t j);
    __host__ __device__ const float& at(size_t i, size_t j) const;

    __host__ __device__ size_t rows() const { return Rows; }
    __host__ __device__ size_t cols() const { return Cols; }
    __host__ __device__ size_t size() const { return Rows * Cols; };

    __host__ __device__ size_t row_rank() const;
    __host__ __device__ size_t col_rank() const;

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

    __host__ __device__ Matrix<Rows - 1, Cols> drop_row(size_t i) const;
    __host__ __device__ Matrix<Rows - 1, Cols> drop_last_row() const;

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


} // namespace mt
} // namespace cumanip


#include "matrix.inl"
#endif