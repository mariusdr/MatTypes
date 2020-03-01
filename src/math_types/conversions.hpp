#ifndef LIBCUMANIP_TYPES_CONVERSIONS_H
#define LIBCUMANIP_TYPES_CONVERSIONS_H

#include "matrix.hpp"
#include "geometry.hpp"


namespace cumanip 
{
namespace mt 
{

__host__ __device__ inline Matrix4f affine(const Matrix3f& rot, const Vector3f& trans);

__host__ __device__ inline Matrix3f rotation(const Matrix4f& affine);

__host__ __device__ inline Vector3f translation(const Matrix4f& affine);

__host__ __device__ inline void from_affine(const Matrix4f& in, Matrix3f& rot, Vector3f& trans);

__host__ __device__ inline mt::Point affine_to_point(const Matrix4f& affine);

__host__ __device__ inline void point_to_trans_rpy(const Point& pt, Vector3f& trans, Vector3f& rpy);

__host__ __device__ inline Matrix4f point_to_affine(const Point& pt);

__host__ __device__ inline Point affine_to_point(const Matrix4f& affine);


//==================================================================================================================//
// Impl                                                                                                             //
//==================================================================================================================//

__host__ __device__ inline 
void point_to_trans_rpy(const Point& pt, Vector3f& trans, Vector3f& rpy)
{
    trans = vec3f(pt.data[0], pt.data[1], pt.data[2]);
    rpy = vec3f(pt.data[3], pt.data[4], pt.data[5]);
}

__host__ __device__ inline 
Matrix4f point_to_affine(const Point& pt)
{
    Vector3f trans;
    Vector3f rpy;
    point_to_trans_rpy(pt, trans, rpy);
    Matrix3f rot = fromRPY(rpy);
    return mt::affine(rot, trans);
}

__host__ __device__ inline
Point affine_to_point(const Matrix4f& affine)
{
    Matrix3f rot;
    Vector3f trans;
    from_affine(affine, rot, trans);
    Vector3f rpy = toRPY(rot);
    return mt::point(trans.data[0], trans.data[1], trans.data[2], rpy.data[0], rpy.data[1], rpy.data[2]);
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
void from_affine(const Matrix4f& in, Matrix3f& rot, Vector3f& trans)
{
    rot = rotation(in);
    trans = translation(in);
}



} // namespace mt
} // namespace cumanip
#endif