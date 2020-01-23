#ifndef TYPES_HPP
#define TYPES_HPP

#ifndef __host__
#define __host__ 
#endif

#ifndef __device__
#define __device__
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>

/**
 *  Vector and Matrix types that can be used on a GPU.
 */

namespace mt 
{

template <size_t Len>
struct Vector;

template <size_t Rows, size_t Cols>
struct Matrix;

using Vector3f = Vector<3>;
using Vector4f = Vector<4>;
using Vector6f = Vector<6>;

using State = Vector<6>;
using Point = Vector<6>;
using Point3 = Vector<3>;

using Matrix3f = Matrix<3, 3>;
using Matrix4f = Matrix<4, 4>;


// ================================================================================= //
// Vector
// ================================================================================= //

template <size_t Len>
struct Vector
{
    float data[Len];
    __host__ __device__ inline Vector() {};
    __host__ __device__ inline explicit Vector(float val); 
    __host__ __device__ inline explicit Vector(float* val);

    __host__ __device__ inline Vector<Len>& operator=(const Vector<Len>& rhs);
    __host__ __device__ inline Vector<Len> operator+(const Vector<Len>& rhs) const;
    __host__ __device__ inline Vector<Len> operator-(const Vector<Len>& rhs) const;
    __host__ __device__ inline Vector<Len> operator*(float) const;
    __host__ __device__ inline Vector<Len> operator/(float) const;

    __host__ __device__ inline size_t size() { return Len; }
    __host__ __device__ inline float length() const;
    __host__ __device__ inline bool approx_equal(const Vector<Len>& rhs, float eps=0.0001) const;
    __host__ __device__ inline float dot(const Vector<Len>& rhs) const;
    __host__ __device__ inline Vector<Len> abs() const;
    __host__ __device__ inline Vector<Len> normalized() const;
};

template <size_t Len>
__host__ __device__ inline Vector<Len> zeros();

template <size_t Len>
__host__ __device__ inline Vector<Len> ones();

__host__ __device__ inline Vector3f vec3f(float x, float y, float z);
__host__ __device__ inline Vector4f vec4f(float x, float y, float z, float w);
__host__ __device__ inline Vector6f vec6f(float a1, float a2, float a3, float a4, float a5, float a6);

__host__ __device__ inline State state(float a1, float a2, float a3, float a4, float a5, float a6);
__host__ __device__ inline Point point(float x, float y, float z, float roll, float pitch, float yaw);
__host__ __device__ inline Point3 point(float x, float y, float z);

__host__ __device__ inline Vector3f get_rpy(const Point& p);
__host__ __device__ inline Point3 get_coords(const Point& p);

__host__ __device__ inline Vector3f cross(const Vector3f& lhs, const Vector3f& rhs);

template <size_t Len>
__host__ __device__ inline float angle(const Vector<Len>& lhs, const Vector<Len>& rhs);

__host__ __device__ inline float deg_to_rad(float deg);
__host__ __device__ inline float rad_to_deg(float rad);

template <size_t Len>
__host__ __device__ inline Vector<Len> deg_to_rad(const Vector<Len>& rhs);

template <size_t Len>
__host__ __device__ inline Vector<Len> rad_to_deg(const Vector<Len>& rhs);

template <size_t Len>
__host__ __device__ inline Vector<Len> operator*(const float& lhs, const Vector<Len>& rhs);

template <size_t Len>
__host__ __device__ inline Vector<Len + 1> cat(const Vector<Len>& lhs, float rhs);

template <size_t Len>
__host__ __device__ inline Vector<Len + 1> cat(float lhs, const Vector<Len>& rhs);

template <size_t Len1, size_t Len2>
__host__ __device__ inline Vector<Len1 + Len2> cat(const Vector<Len1>& rhs, const Vector<Len2>& lhs);

template <size_t Len>
__host__ __device__ inline void printVec(const Vector<Len>& vec, const char* fmt = "%f ");

template <size_t Len>
__host__ inline std::ostream& operator<<(std::ostream&, const Vector<Len>& vec);


// ================================================================================= //
// Matrix
// ================================================================================= //

template <size_t Rows, size_t Cols>
struct Matrix
{
    float data[Rows * Cols];
    __host__ __device__ inline Matrix() {};
    __host__ __device__ inline explicit Matrix(float val);
    __host__ __device__ inline explicit Matrix(float* val);

    __host__ __device__ inline size_t rows() { return Rows; }
    __host__ __device__ inline size_t cols() { return Cols; }

    __host__ __device__ inline Matrix<Rows, Cols>& operator=(const Matrix<Rows, Cols>& rhs);

    __host__ __device__ inline void set_row(size_t i, const Vector<Cols> &row);
    __host__ __device__ inline void set_col(size_t j, const Vector<Rows>& col);

    __host__ __device__ inline Vector<Cols> get_row(size_t i) const;
    __host__ __device__ inline Vector<Rows> get_col(size_t j) const;

    __host__ __device__ inline Matrix<Cols, Rows> transpose() const;

    __host__ __device__ inline float trace() const;

    __host__ __device__ inline bool approx_equal(const Matrix<Rows, Cols>& rhs, float eps=0.0001);
};

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
__host__ __device__ inline Matrix4f affine(const Matrix3f& rot, const Vector3f& trans);
__host__ __device__ inline Matrix3f rotation(const Matrix4f& affine);
__host__ __device__ inline Vector3f translation(const Matrix4f& affine);

__host__ __device__ inline Matrix3f from_roll(float a);
__host__ __device__ inline Matrix3f from_pitch(float a);
__host__ __device__ inline Matrix3f from_yaw(float a);
__host__ __device__ inline Matrix3f fromRPY(float r, float p, float y);
__host__ __device__ inline Matrix3f fromRPY(Vector3f rpy);
__host__ __device__ inline Matrix3f fromYPR(float y, float p, float r);
__host__ __device__ inline Matrix3f fromYPR(Vector3f ypr);

__host__ __device__ inline Vector3f toRPY(const Matrix3f& rot, unsigned int solution_number = 1);

template <size_t Rows, size_t Cols>
__host__ __device__ inline void printMat(const Matrix<Rows, Cols>& mat, const char* fmt = "%f ");

template <size_t Rows, size_t Cols>
__host__ inline std::ostream& operator<<(std::ostream& os, const Matrix<Rows, Cols>& mat);







// ================================================================================= //
// Implementations
// ================================================================================= //

template <size_t Len> 
__host__ __device__ inline
Vector<Len>::Vector(float val)
{
    for (size_t i = 0; i < Len; ++i) 
        data[i] = val;
}

template <size_t Len> 
__host__ __device__ inline
Vector<Len>& Vector<Len>::operator=(const Vector<Len>& rhs)
{
    for (size_t i = 0; i < Len; ++i)
        data[i] = rhs.data[i];
    return *this;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::operator+(const Vector<Len>& rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] + rhs.data[i];
    return res;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::operator-(const Vector<Len>& rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] - rhs.data[i];
    return res;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::operator*(float rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] * rhs;
    return res;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::operator/(float rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] / rhs;
    return res;
}

template <size_t Len>
__host__ __device__ inline
void printVec(const Vector<Len>& vec, const char* fmt)
{
    for (size_t i = 0; i < Len; ++i)
        printf(fmt, vec.data[i]);
    printf("\n");
}

template <size_t Len>
__host__ inline
std::ostream& operator<<(std::ostream& os, const Vector<Len>& vec)
{
    for (size_t i = 0; i < Len; ++i)
        os << vec.data[i] << " ";
    os << "\n";
    return os;
}

template <size_t Len>
__host__ __device__ inline
bool Vector<Len>::approx_equal(const Vector<Len>& rhs, float eps) const
{
    for (size_t i = 0; i < Len; ++i)
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

template <size_t Len>
__host__ __device__ inline
float Vector<Len>::dot(const Vector<Len>& rhs) const 
{
    float sum = 0;
    for (size_t i = 0; i < Len; ++i)
    {
        sum += data[i] * rhs.data[i];
    }
    return sum;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::abs() const 
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
    {
        res.data[i] = fabs(data[i]);
    }
    return res;
}

template <size_t Len>
__host__ __device__ inline
Vector<Len> Vector<Len>::normalized() const 
{
    return *this / length();
}

template <size_t Len>
__host__ __device__ inline
float Vector<Len>::length() const 
{
    float acc = 0.f;
    for (size_t i = 0; i < Len; ++i)
    {
        acc += data[i] * data[i];
    }
    return sqrt(acc); 
}

__host__ __device__ inline 
Vector3f cross(const Vector3f& lhs, const Vector3f& rhs)
{
    Vector3f res;
    res.data[0] = lhs.data[1] * rhs.data[2] - lhs.data[2] * rhs.data[1];
    res.data[1] = lhs.data[2] * rhs.data[0] - lhs.data[0] * rhs.data[2];
    res.data[2] = lhs.data[0] * rhs.data[1] - lhs.data[1] * rhs.data[0];
    return res;
}

__host__ __device__ inline 
Vector3f vec3f(float x, float y, float z)
{
    Vector3f res;
    res.data[0] = x;
    res.data[1] = y;
    res.data[2] = z;
    return res;
}

__host__ __device__ inline 
Vector4f vec4f(float x, float y, float z, float w)
{
    Vector4f res;
    res.data[0] = x;
    res.data[1] = y;
    res.data[2] = z;
    res.data[3] = w;
    return res;

}
__host__ __device__ inline 
Vector6f vec6f(float a1, float a2, float a3, float a4, float a5, float a6)
{
    Vector6f res;
    res.data[0] = a1;
    res.data[1] = a2;
    res.data[2] = a3;
    res.data[3] = a4;
    res.data[4] = a5;
    res.data[5] = a6;
    return res;
}

__host__ __device__ inline 
State state(float a1, float a2, float a3, float a4, float a5, float a6)
{
    return vec6f(a1, a2, a3, a4, a5, a6);
}

__host__ __device__ inline 
Point point(float x, float y, float z, float roll, float pitch, float yaw)
{
    return vec6f(x, y, z, roll, pitch, yaw);
}

__host__ __device__ inline 
Point3 point(float x, float y, float z)
{
    return vec3f(x, y, z);
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len> zeros()
{
    return Vector<Len>(0.f);
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len> ones()
{
    return Vector<Len>(1.f);
}

__host__ __device__ inline
Vector3f get_rpy(const Point& p)
{
    return vec3f(p.data[3], p.data[4], p.data[5]);
}

__host__ __device__ inline
Point3 get_coords(const Point& p)
{
    return point(p.data[0], p.data[1], p.data[2]);
}

template <size_t Len>
__host__ __device__ inline float angle(const Vector<Len>& lhs, const Vector<Len>& rhs)
{
    float dp = lhs.dot(rhs);
    dp /= lhs.length();
    dp /= rhs.length();
    return acos(dp);
}

__host__ __device__ inline 
float deg_to_rad(float deg)
{
    return deg * (M_PI / 180.f);
}

__host__ __device__ inline 
float rad_to_deg(float rad)
{
    return rad * (180.f / M_PI);
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len> deg_to_rad(const Vector<Len>& rhs)
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
    {
        res.data[i] = deg_to_rad(rhs.data[i]);
    }
    return res;
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len> rad_to_deg(const Vector<Len>& rhs)
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
    {
        res.data[i] = rad_to_deg(rhs.data[i]);
    }
    return res;
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len> operator*(const float& lhs, const Vector<Len>& rhs)
{
    return rhs * lhs;
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
__host__ __device__ inline
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
__host__ __device__ inline
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
__host__ __device__ inline
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
__host__ __device__ inline
void Matrix<Rows, Cols>::set_row(size_t i, const Vector<Cols>& row)
{
    for (size_t j = 0; j < Cols; ++j)
    {
        data[i * Cols + j] = row.data[j];
    }
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline
void Matrix<Rows, Cols>::set_col(size_t j, const Vector<Rows>& col)
{
    for (size_t i = 0; i < Rows; ++i)
    {
        data[i * Cols + j] = col.data[i];
    }
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline
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
__host__ __device__ inline
Vector<Rows> Matrix<Rows, Cols>::get_col(size_t j) const 
{
    Vector<Rows> v;
    for (size_t i = 0; i < Rows; ++i)
    {
        v.data[i] = data[i * Cols + j];
    }
    return v;
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len + 1> cat(const Vector<Len>& lhs, float rhs)
{
    Vector<Len + 1> v;
    for (size_t i = 0; i < Len; ++i)
    {
        v.data[i] = lhs.data[i];
    }
    v.data[Len] = rhs;
    return v;
}

template <size_t Len>
__host__ __device__ inline 
Vector<Len + 1> cat(float lhs, const Vector<Len>& rhs)
{
    Vector<Len + 1> v;
    v.data[0] = lhs;
    for (size_t i = 1; i < Len + 1; ++i)
    {
        v.data[i] = rhs.data[i];
    }
    return v;
}

template <size_t Len1, size_t Len2>
__host__ __device__ inline 
Vector<Len1 + Len2> cat(const Vector<Len1>& rhs, const Vector<Len2>& lhs)
{
    Vector<Len1 + Len2> v;
    for (size_t i = 0; i < Len1; ++i)
    {
        v.data[i] = rhs.data[i];
    }
    for (size_t i = 0; i < Len2; ++i)
    {
        v.data[Len1 + i] = lhs.data[i];
    }
    return v;
}

__host__ __device__ inline 
Matrix3f rotation(const Matrix4f& affine)
{
    Vector4f r1 = affine.get_row(0);
    Vector4f r2 = affine.get_row(1);
    Vector4f r3 = affine.get_row(2);
    return mat3f_rows(
        vec3f(r1.data[0], r1.data[1], r1.data[2]),
        vec3f(r2.data[0], r2.data[1], r2.data[2]),
        vec3f(r3.data[0], r3.data[1], r3.data[2])
    );
}

__host__ __device__ inline 
Vector3f translation(const Matrix4f& affine)
{
    Vector4f r1 = affine.get_row(0);
    Vector4f r2 = affine.get_row(1);
    Vector4f r3 = affine.get_row(2);
    return vec3f(r1.data[3], r2.data[3], r3.data[3]);
}

__host__ __device__ inline 
Matrix4f affine(const Matrix3f& rot, const Vector3f& trans)
{
    Vector4f r1 = cat(rot.get_row(0), trans.data[0]);
    Vector4f r2 = cat(rot.get_row(1), trans.data[1]);
    Vector4f r3 = cat(rot.get_row(2), trans.data[2]);
    Vector4f r4 = vec4f(0.f, 0.f, 0.f, 1.f);
    return mat4f_rows(r1, r2, r3, r4);
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

template<size_t Len>
__host__ __device__ inline 
Vector<Len>::Vector(float* val)
{
    for (size_t i = 0; i < Len; ++i)
    {
        data[i] = val[i];
    }
}

template<size_t Rows, size_t Cols>
__host__ __device__ inline 
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
Matrix3f from_roll(float a)
{
    return mat3f(
        1, 0, 0,
        0, cos(a), -sin(a),
        0, sin(a), cos(a));
}

__host__ __device__ inline 
Matrix3f from_pitch(float a)
{
    return mat3f(
        cos(a), 0, sin(a),
        0, 1, 0,
        -sin(a), 0, cos(a));
}

__host__ __device__ inline 
Matrix3f from_yaw(float a)
{
    return mat3f(
        cos(a), -sin(a), 0,
        sin(a), cos(a), 0,
        0, 0, 1
    );
}

__host__ __device__ inline 
Matrix3f fromRPY(float r, float p, float y)
{
    return from_yaw(y) * from_pitch(p) * from_roll(r);
}

__host__ __device__ inline 
Matrix3f fromRPY(Vector3f rpy)
{
    return fromRPY(rpy.data[0], rpy.data[1], rpy.data[2]);
}

__host__ __device__ inline 
Matrix3f fromYPR(float y, float p, float r)
{
    return from_roll(r) * from_pitch(p) * from_yaw(y);
}

__host__ __device__ inline 
Matrix3f fromYPR(Vector3f ypr)
{
    return fromYPR(ypr.data[0], ypr.data[1], ypr.data[2]);
}

__host__ __device__ inline
void coeffs_vec3f(const Vector3f& vec, float& x, float& y, float& z)
{
    x = vec.data[0];
    y = vec.data[1];
    z = vec.data[2];
}

__host__ __device__ inline
void coeffs_mat3f(const Matrix3f& mat, float& a11, float& a12, float& a13, float& a21, float& a22, float& a23, float& a31, float& a32, float& a33)
{
    coeffs_vec3f(mat.get_row(0), a11, a12, a13);
    coeffs_vec3f(mat.get_row(1), a21, a22, a23);
    coeffs_vec3f(mat.get_row(3), a31, a32, a33);
}


// from https://github.com/fzi-forschungszentrum-informatik/gpu-voxels/blob/master/packages/gpu_voxels/src/gpu_voxels/helpers/cuda_matrices.h
__host__ __device__ inline 
Vector3f toRPY(const Matrix3f& rot, unsigned int solution_number)
{
    float x1, y1, z1;
    float x2, y2, z2;

    const float a11 = rot.data[0 * 3 + 0];
    const float a12 = rot.data[0 * 3 + 1];
    const float a13 = rot.data[0 * 3 + 2];
    
    const float a21 = rot.data[1 * 3 + 0];
    const float a22 = rot.data[1 * 3 + 1];
    const float a23 = rot.data[1 * 3 + 2];

    const float a31 = rot.data[2 * 3 + 0];
    const float a32 = rot.data[2 * 3 + 1];
    const float a33 = rot.data[2 * 3 + 2];

    // Check that pitch is not at a singularity
    if (1.0 - fabs(a31) < 0.00001)
    {
        z1 = 0;
        z2 = 0;

        // From difference of angles formula
        if (a31 < 0) //gimbal locked down
        {
            float delta = atan2(a12, a13);
            y1 = M_PI_2;
            y2 = M_PI_2;
            x1 = delta;
            x2 = delta;
        }
        else // gimbal locked up
        {
            float delta = atan2(-a12, -a13);
            y1 = -M_PI_2;
            y2 = -M_PI_2;
            x1 = delta;
            x2 = delta;
        }
    }
    else
    {
        y1 = -asin(a31);
        y2 = M_PI - y1;

        x1 = atan2(a32 / cos(y1), a33 / cos(y1));
        x2 = atan2(a32 / cos(y2), a33 / cos(y2));

        z1 = atan2(a21 / cos(y1), a11 / cos(y1));
        z2 = atan2(a21 / cos(y2), a11 / cos(y2));
    }

    return (solution_number == 1) ? vec3f(x1, y1, z1) : vec3f(x2, y2, z2);
}

template <size_t Rows, size_t Cols>
__host__ __device__ inline 
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


} // namespace mt


#endif