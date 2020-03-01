#ifndef LIBCUMANIP_TRAJECTORIES_INVERSE_KINEMATICS_H
#define LIBCUMANIP_TRAJECTORIES_INVERSE_KINEMATICS_H

#include "kinematics.hpp"
#include "typedefs.h"

#include <cstring>
#include <iostream>

//!
#include <vector>
#include <tuple>
//!

namespace cumanip
{

struct IKSolution 
{
    int num_solutions;
    mt::Matrix<8, 6> states;
    unsigned char status[8];
};

__host__ inline
std::ostream& operator<<(std::ostream& os, const IKSolution& sol);


__host__ inline 
void run_inverse_kin(Mat4_DevIter in_first, Mat4_DevIter in_last, IKSolution_DevIter out_first);




/////////////////////////////////////////////////////////////////////////////////////////////////

struct RunIKOp
{
    __host__ __device__
    IKSolution operator()(const mt::Matrix4f& transform)
    {
        URKInverseKinematics ik;
        int n = ik.solve(transform);

        IKSolution sol;
        sol.num_solutions = n;
        sol.states = ik.get_solutions();
        memcpy(sol.status, ik.get_status(), 8 * sizeof(unsigned char));
        
        return sol;
    }
};

__host__ inline 
void run_inverse_kin(Mat4_DevIter in_first, Mat4_DevIter in_last, IKSolution_DevIter out_first)
{
    RunIKOp ikOp;
    //!
    std::transform(in_first, in_last, out_first, ikOp);
    //!
}

__host__ inline
std::ostream& operator<<(std::ostream& os, const IKSolution& sol)
{
    os << sol.states;
    return os;
}






} // namespace cumanip
#endif