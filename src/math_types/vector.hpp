#ifndef LIBCUMANIP_TYPES_VECTOR_H
#define LIBCUMANIP_TYPES_VECTOR_H

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


template <size_t Len>
struct Vector
{
    __host__ __device__ Vector() {};
    __host__ __device__ explicit Vector(float val); 
    __host__ __device__ explicit Vector(float* val);
    __host__ __device__ Vector(const Vector<Len>& rhs);

    __host__ Vector(std::initializer_list<float> l);

    __host__ __device__ float& at(size_t i);
    __host__ __device__ const float& at(size_t i) const;

    __host__ __device__ size_t size() { return Len; }

    __host__ __device__ bool is_zero() const;

    __host__ __device__ Vector<Len>& operator=(const Vector<Len>& rhs);
    __host__ __device__ Vector<Len> operator+(const Vector<Len>& rhs) const;
    __host__ __device__ Vector<Len> operator-(const Vector<Len>& rhs) const;
    __host__ __device__ Vector<Len> operator*(float) const;
    __host__ __device__ Vector<Len> operator/(float) const;

    __host__ __device__ float dot(const Vector<Len>& rhs) const;
    __host__ __device__ bool approx_equal(const Vector<Len>& rhs, float eps=0.000001) const;
    
    __host__ __device__ float length() const;
    __host__ __device__ Vector<Len> abs() const;
    __host__ __device__ Vector<Len> normalized() const;

    __host__ __device__ float max() const;
    __host__ __device__ float min() const;
    __host__ __device__ float sum() const;

    __host__ __device__ float max(size_t& idx) const;
    __host__ __device__ float min(size_t& idx) const;

    float data[Len];
};

template <size_t Len>
__host__ __device__ inline Vector<Len> operator*(const float& lhs, const Vector<Len>& rhs);

template <size_t Len>
__host__ inline std::ostream& operator<<(std::ostream&, const Vector<Len>& vec);


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


template <size_t Len>
__host__ __device__ inline Vector<Len + 1> cat(const Vector<Len>& lhs, float rhs);

template <size_t Len>
__host__ __device__ inline Vector<Len + 1> cat(float lhs, const Vector<Len>& rhs);

template <size_t Len1, size_t Len2>
__host__ __device__ inline Vector<Len1 + Len2> cat(const Vector<Len1>& rhs, const Vector<Len2>& lhs);

template <size_t Len>
__host__ __device__ inline void printVec(const Vector<Len>& vec, const char* fmt = "%f ");



} // namespace mt
} // namespace cumanip

#include "vector.inl"
#endif
