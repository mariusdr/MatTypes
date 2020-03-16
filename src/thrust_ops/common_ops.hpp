#ifndef LIBCUMANIP_COMMON_OPS_H
#define LIBCUMANIP_COMMON_OPS_H

#include <cstring>

//!
#include "../math_types.hpp"
#include "../kinematics.hpp"
//!

namespace cumanip
{

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

struct PointToXYZ
{
    __host__ __device__ 
    mt::Vector3f operator()(const mt::Point& point)
    {
        return mt::get_coords(point);
    }
};

struct PointToRPY
{
    __host__ __device__
    mt::Vector3f operator()(const mt::Point& point)
    {
        return mt::get_rpy(point);
    }
};

struct ApplyRightTransform
{
    ApplyRightTransform(mt::Matrix4f t): transform(t) 
    {}

    __host__ __device__
    mt::Matrix4f operator()(const mt::Matrix4f& inp)
    {
        return inp * transform;
    }
    
    mt::Matrix4f transform;
};

struct ApplyLeftTransform
{
    ApplyLeftTransform(mt::Matrix4f t): transform(t) 
    {}

    __host__ __device__
    mt::Matrix4f operator()(const mt::Matrix4f& inp)
    {
        return transform * inp;
    }
    
    mt::Matrix4f transform;
};

template <size_t Rows, size_t Cols>
struct MinMat
{
    __host__ __device__
    float operator()(const mt::Matrix<Rows, Cols>& mat)
    {
        return mat.min();
    }
};

template <size_t Rows, size_t Cols>
struct MaxMat
{
    __host__ __device__
    float operator()(const mt::Matrix<Rows, Cols>& mat)
    {
        return mat.max();
    }
};

template <size_t Rows, size_t Cols>
struct SumMat
{
    __host__ __device__
    float operator()(const mt::Matrix<Rows, Cols>& mat)
    {
        return mat.sum();
    }
};

template <size_t Len>
struct MinVec 
{
    __host__ __device__
    float operator()(const mt::Vector<Len>& vec)
    {
        return vec.min();
    }
};

template <size_t Len>
struct MaxVec 
{
    __host__ __device__
    float operator()(const mt::Vector<Len>& vec)
    {
        return vec.max();
    }
};

template <size_t Len>
struct SumVec 
{
    __host__ __device__
    float operator()(const mt::Vector<Len>& vec)
    {
        return vec.sum();
    }
};



// template <size_t NumSamples>
// struct GenerateRandomPoses
// {
//     __host__ __device__ 
//     mt::Matrix<NumSamples, 3> operator()(const unsigned int n) const
//     {
//         thrust::minstd_rand rng;
//         thrust::uniform_real_distribution<float> dist(0.f, 2 * M_PI);
//         rng.discard(n * 3 * NumSamples);

//         mt::Matrix<NumSamples, 3> samples;
//         for (size_t i = 0; i < NumSamples; ++i)
//         {
//             float r = dist(rng);
//             float p = dist(rng);
//             float y = dist(rng);

//             mt::Vector3f rpy = mt::vec3f(r, p, y);
//             samples.set_row(i, rpy);
//         }
//         return samples;
//     }
// };

} // namespace
#endif