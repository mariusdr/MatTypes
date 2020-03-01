#ifndef LIBCUMANIP_TRAJECTORIES_FORWARD_KINEMATICS_H
#define LIBCUMANIP_TRAJECTORIES_FORWARD_KINEMATICS_H

#include "kinematics.hpp"
#include "typedefs.h"

//!
#include <vector>
#include <algorithm>
//!


namespace cumanip 
{


__host__ inline 
void run_forward_kin(State_DevIter in_first, State_DevIter in_last, Mat4_DevIter out_first);

__host__ inline 
void run_forward_kin(State_HostIter in_first, State_HostIter in_last, Mat4_HostIter out_first);




///////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct RunFKOp
{
    __host__ __device__ 
    mt::Matrix4f operator()(const mt::State& state)
    {
        URKForwardKinematics fk;
        fk.solve(state);
        return fk.transform_base_to_ee(); 
    }
};

__host__ inline 
void run_forward_kin(State_DevIter in_first, State_DevIter in_last, Mat4_DevIter out_first)
{
    RunFKOp fkOp;
    //!
    std::transform(in_first, in_last, out_first, fkOp);
    //!
    // CUDA_ASSERT(cudaGetLastError());
}

__host__ inline 
void run_forward_kin(State_HostIter in_first, State_HostIter in_last, Mat4_HostIter out_first)
{
    RunFKOp fkOp;
    //!
    std::transform(in_first, in_last, out_first, fkOp);
    //!
    // CUDA_ASSERT(cudaGetLastError());
}



}
#endif