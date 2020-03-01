#ifndef LIBCUMANIP_EXTRACT_TRAJECTORIES_BY_STATUS_H
#define LIBCUMANIP_EXTRACT_TRAJECTORIES_BY_STATUS_H

#include "kinematics.hpp"
#include "inverse_kinematics_op.h"
#include "typedefs.h"

//!
#include <vector>
#include <algorithm>
//!


namespace cumanip 
{


__host__ inline 
void extract_trajectory_by_status(IKSolution_DevIter inp_begin, IKSolution_DevIter inp_end, 
                                  State_DevIter outp_begin, unsigned char target_status);


__host__ inline 
std::vector<StateTrajectoryDev> get_all_trajectories_by_status(IKSolution_DevIter inp_begin, 
                                                               IKSolution_DevIter inp_end);



/////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct ExtractTrajectoryByStatus 
{
    __device__ __host__
    ExtractTrajectoryByStatus(unsigned char target): target_status(target)
    {}

    __device__ __host__ 
    mt::State operator()(const IKSolution& sol)
    {
        mt::State tar(0.f);

        int n = sol.num_solutions;
        for (int i = 0; i < n; ++i)
        {
            unsigned char st = sol.status[i];
            tar = (target_status == st) ? sol.states.get_row(i) : tar;
        }

        return tar;
    }

    unsigned char target_status;
};


struct SolutionHasStatus 
{
    __device__ __host__
    SolutionHasStatus(unsigned char target): target_status(target)
    {}

    __device__ __host__
    bool operator()(const IKSolution& sol)
    {
        bool res = false;

        int n = sol.num_solutions;
        for (int i = 0; i < n; ++i)
        {
            unsigned char st = sol.status[i];
            res |= (st == target_status);
        }

        return res;
    }

    unsigned char target_status;
};

struct IsTruePred
{
    __host__ __device__
    bool operator()(bool x)
    {
        return x == true;
    }
};


__host__ inline 
void extract_trajectory_by_status(IKSolution_DevIter inp_begin, IKSolution_DevIter inp_end, 
                                  State_DevIter outp_begin, unsigned char target_status)
{
    ExtractTrajectoryByStatus ex_op(target_status);
    //!
    std::transform(inp_begin, inp_end, outp_begin, ex_op);
    //!
}


__host__ inline 
std::vector<StateTrajectoryDev> get_all_trajectories_by_status(IKSolution_DevIter inp_begin, 
                                                               IKSolution_DevIter inp_end)
{
    //!
    std::vector<StateTrajectoryDev> trajectories;
    //!

    for (int i = 0; i < 8; ++i)
    {
        unsigned char tar = (unsigned char) i;

        SolutionHasStatus has_st_op(tar);
        IsTruePred pred;
        
        //! 
        size_t len = std::distance(inp_begin, inp_end);
        std::vector<bool> vs(len);
        std::transform(inp_begin, inp_end, vs.begin(), has_st_op);
        bool has_status = std::all_of(vs.begin(), vs.end(), pred);
        //!

        if (has_status)
        {
            //! 
            std::vector<mt::State> states(len);
            //!

            extract_trajectory_by_status(inp_begin, inp_end, states.begin(), tar);
            trajectories.push_back(states);
        }
    }

    return trajectories;
}





} // namespace
#endif