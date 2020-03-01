#ifndef LIBCUMANIP_COMMON_OPS_H
#define LIBCUMANIP_COMMON_OPS_H

#include "kinematics.hpp"
#include "typedefs.h"

#include <cstring>

//!
#include <vector>
#include <algorithm>
//!

namespace cumanip
{


__host__ inline 
void convert_point_to_affine(Point_DevIter in_first, Point_DevIter in_last, Mat4_DevIter out_first);

__host__ inline 
void convert_point_to_affine(Point_HostIter in_first, Point_HostIter in_last, Mat4_HostIter out_first);


__host__ inline 
void convert_affine_to_point(Mat4_DevIter in_first, Mat4_DevIter in_last, Point_DevIter out_first);

__host__ inline 
void convert_affine_to_point(Mat4_HostIter in_first, Mat4_HostIter in_last, Point_HostIter out_first);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




struct PointToAffine
{
    __device__ __host__ 
    mt::Matrix4f operator()(const mt::Point& pt)
    {
        return mt::point_to_affine(pt);
    }
};

struct AffineToPoint
{
    __device__ __host__
    mt::Point operator()(const mt::Matrix4f& mat)
    {
        return mt::affine_to_point(mat);
    }
};



__device__ __host__ inline
void convert_point_to_affine(Point_DevIter in_first, Point_DevIter in_last, Mat4_DevIter out_first)
{
    PointToAffine op;
    //!
    std::transform(in_first, in_last, out_first, op);
    //!
}

__device__ __host__ inline
void convert_point_to_affine(Point_HostIter in_first, Point_HostIter in_last, Mat4_HostIter out_first)
{
    PointToAffine op;
    //!
    std::transform(in_first, in_last, out_first, op);
    //!
}




__device__ __host__ inline
void convert_affine_to_point(Mat4_DevIter in_first, Mat4_DevIter in_last, Point_DevIter out_first)
{
    AffineToPoint op;
    //!
    std::transform(in_first, in_last, out_first, op);
    //!
}

__device__ __host__ inline
void convert_affine_to_point(Mat4_HostIter in_first, Mat4_HostIter in_last, Point_HostIter out_first)
{
    AffineToPoint op;
    //!
    std::transform(in_first, in_last, out_first, op);
    //!
}


} // namespace
#endif