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

#ifndef __host__ 
#define __host__ 
#endif
#ifndef __device__ 
#define __device__ 
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

    __host__ __device__ size_t size() { return Len; }
    
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



//==================================================================================================================//
// Vector Class Impl                                                                                                //
//==================================================================================================================//


template<size_t Len>
__host__ __device__ 
Vector<Len>::Vector(float* val)
{
    for (size_t i = 0; i < Len; ++i)
    {
        data[i] = val[i];
    }
}

template <size_t Len> 
__host__ __device__ 
float Vector<Len>::max() const 
{
    float v = data[0];
    for (size_t i = 0; i < Len; ++i)
    {
        if (data[i] > v)
        {
            v = data[i];
        }
    }
    return v;
}

template <size_t Len> 
__host__ __device__ 
float Vector<Len>::min() const 
{
    float v = data[0];
    for (size_t i = 0; i < Len; ++i)
    {
        if (data[i] < v)
        {
            v = data[i];
        }
    }
    return v;
}

template <size_t Len> 
__host__ __device__ 
float Vector<Len>::max(size_t& idx) const 
{
    float v = data[0];
    for (size_t i = 0; i < Len; ++i)
    {
        if (data[i] > v)
        {
            v = data[i];
            idx = i; 
        }
    }
    return v;
}

template <size_t Len> 
__host__ __device__ 
float Vector<Len>::min(size_t& idx) const 
{
    float v = data[0];
    for (size_t i = 0; i < Len; ++i)
    {
        if (data[i] < v)
        {
            v = data[i];
            idx = i;
        }
    }
    return v;
}


template <size_t Len> 
__host__ __device__ 
Vector<Len>::Vector(float val)
{
    for (size_t i = 0; i < Len; ++i) 
        data[i] = val;
}

template <size_t Len> 
__host__ __device__ 
Vector<Len>::Vector(const Vector<Len>& rhs)
{
    for (size_t i = 0; i < Len; ++i)
    {
        data[i] = rhs.data[i];
    }
}


template <size_t Len> 
__host__ __device__ 
Vector<Len>& Vector<Len>::operator=(const Vector<Len>& rhs)
{
    if (this == &rhs)
    {
        return *this;
    }

    for (size_t i = 0; i < Len; ++i)
    {
        data[i] = rhs.data[i];
    }
    return *this;
}

template <size_t Len>
__host__ __device__ 
Vector<Len> Vector<Len>::operator+(const Vector<Len>& rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] + rhs.data[i];
    return res;
}

template <size_t Len>
__host__ __device__ 
Vector<Len> Vector<Len>::operator-(const Vector<Len>& rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] - rhs.data[i];
    return res;
}

template <size_t Len>
__host__ __device__ 
Vector<Len> Vector<Len>::operator*(float rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] * rhs;
    return res;
}

template <size_t Len>
__host__ __device__ 
Vector<Len> Vector<Len>::operator/(float rhs) const
{
    Vector<Len> res;
    for (size_t i = 0; i < Len; ++i)
        res.data[i] = this->data[i] / rhs;
    return res;
}

template <size_t Len>
__host__ __device__ 
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


template <size_t Len>
__host__ __device__ 
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
__host__ __device__ 
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
__host__ __device__ 
Vector<Len> Vector<Len>::normalized() const 
{
    return *this / length();
}

template <size_t Len>
__host__ __device__ 
float Vector<Len>::length() const 
{
    float acc = 0.f;
    for (size_t i = 0; i < Len; ++i)
    {
        acc += data[i] * data[i];
    }
    return sqrt(acc); 
}


//==================================================================================================================//
// Vector Functions Impl                                                                                            //
//==================================================================================================================//

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

template <size_t Len>
__host__ __device__ inline 
Vector<Len> operator*(const float& lhs, const Vector<Len>& rhs)
{
    return rhs * lhs;
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
void coeffs_vec3f(const Vector3f& vec, float& x, float& y, float& z)
{
    x = vec.data[0];
    y = vec.data[1];
    z = vec.data[2];
}

} // namespace mt
} // namespace cumanip
#endif