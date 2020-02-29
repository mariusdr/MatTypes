#ifndef LIBCUMANIP_GEOMETRY_H
#define LIBCUMANIP_GEOMETRY_H 

#include "vector.hpp"
#include "matrix.hpp"

namespace cumanip
{
namespace mt 
{

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

__host__ __device__ inline State state_from_deg(float a1, float a2, float a3, float a4, float a5, float a6);

//==================================================================================================================//
// Impl                                                                                                             //
//==================================================================================================================//


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
__host__ __device__ inline 
float angle(const Vector<Len>& lhs, const Vector<Len>& rhs)
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
State state_from_deg(float a1, float a2, float a3, float a4, float a5, float a6)
{
    return deg_to_rad(state(a1, a2, a3, a4, a5, a6));
}




} // namespace
} // namespace
#endif