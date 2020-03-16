#ifndef LIBCUMANIP_PLANNING_OPS_H
#define LIBCUMANIP_PLANNING_OPS_H

#include <cstring>

//!
#include "../math_types.hpp"
#include "../kinematics.hpp"

#include "../cuda_utils.h"
//!


namespace cumanip
{

template <size_t NumStates>
struct ComputeDistanceMatrix
{
    using InMatrix = mt::Matrix<NumStates, 6>;
    using OutMatrix = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    ComputeDistanceMatrix(mt::Matrix<6, 6> joint_weights): joint_weights(joint_weights)
    {}
    
    __device__ __host__ 
    ComputeDistanceMatrix(): joint_weights(mt::identity<6, 6>())
    {}

    __device__ __host__ 
    OutMatrix operator()(const InMatrix& fst, const InMatrix& snd)
    {
        OutMatrix dm(infty());

        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                const mt::State& si = fst.get_row(i);
                const mt::State& sj = snd.get_row(j);
                dm.at(i, j) = mt::distance(joint_weights, si, sj);
            }
        }
        return dm;
    }
    
    // __device__ __host__
    // OutMatrix operator()(const thrust::tuple<InMatrix, InMatrix>& p)
    // {
    //     return (*this)(thrust::get<0>(p), thrust::get<1>(p));
    // }

    mt::Matrix<6, 6> joint_weights;
};


template <size_t NumStates>
struct ComputeTransitionMatrix
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    ComputeTransitionMatrix(float md): max_dist(md)
    {}

    __device__ __host__
    MatrixN operator()(const MatrixN& distance_matrix)
    {
        MatrixN out(0.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                const float dist = distance_matrix.at(i, j);
                out.at(i, j) = (dist <= max_dist) ? 1.f : 0.f;
            }
        }
        return out;
    }
    
    float max_dist;
};


template <size_t NumStates>
struct CombineTransitionMatricies
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    MatrixN operator()(const MatrixN& fst, const MatrixN& snd)
    {
        MatrixN out(0.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                float v = 0.f;
                for (size_t k = 0; k < NumStates; ++k)
                {
                    const float& x_ik = fst.at(i, k);
                    const float& x_kj = snd.at(k, j);
                    const float& m = x_ik < x_kj ? x_ik : x_kj;
                    v = v > m ? v : m;
                }
                out.at(i, j) = v;
            }
        }
        return out;
    }
};



} // namespace
#endif