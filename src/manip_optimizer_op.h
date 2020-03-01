#ifndef LIBCUMANIP_MANIPULABILITY_OPTIMIZER_H
#define LIBCUMANIP_MANIPULABILITY_OPTIMIZER_H

#include "kinematics.hpp"

#include "inverse_kinematics_op.h"

//!
#include <vector>
#include <algorithm>
//!

namespace cumanip
{

using TMatrix = mt::Matrix<8, 8>;




/////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct ComputeTransitionMatrixOp
{
    __device__ __host__ 
    ComputeTransitionMatrixOp(float md): max_dist(md), dweights(mt::identity<6, 6>())
    {}

    __device__ __host__ 
    ComputeTransitionMatrixOp(float md, mt::Matrix<6, 6> weights): max_dist(md), dweights(weights)
    {}

    __device__ __host__
    TMatrix operator()(const IKSolution& from, const IKSolution& to)
    {
        TMatrix out(0.f);

        for (size_t i = 0; i < from.num_solutions; ++i)
        {
            for (size_t j = 0; j < to.num_solutions; ++j)
            {
                mt::State s_from = from.states.get_row(i);
                mt::State s_to = to.states.get_row(j);
                float dist = mt::distance(dweights, s_from, s_to);
                out.data[i * 8 + j] = (dist < max_dist) ? 1.f : 0.f;
            }
        }
        return out;
    }

    // __device__ __host__
    // TMatrix operator()(const thrust::tuple<IKSolution, IKSolution>& p)
    // {
    //     return (*this)(thrust::get<0>(p), thrust::get<1>(p));
    // }
    
    float max_dist;
    mt::Matrix<6, 6> dweights;
};








} // namespace
#endif