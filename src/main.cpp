#include <iostream>
#include <iomanip>
#include <cstdlib>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include "math_types.hpp"
#include "kinematics.hpp"
#include "forward_kinematics_op.h"
#include "inverse_kinematics_op.h"
#include "common_op.h"
#include "extract_traj_by_status.h"


#include <vector>

int main()
{
    using namespace cumanip;

    std::vector<mt::State> inp_states;
    for (size_t i = 0; i < 50; ++i)
    {
        mt::State s = mt::state(0.12, 1.23, 0.23, 0.44, -0.22, 1.31);
        inp_states.push_back(s);
    }

    std::vector<mt::Matrix4f> poses(inp_states.size());
    run_forward_kin(inp_states.begin(), inp_states.end(), poses.begin());

    std::vector<IKSolution> ik_solutions(poses.size());
    run_inverse_kin(poses.begin(), poses.end(), ik_solutions.begin());

    std::vector<std::vector<mt::State>> trajectories = get_all_trajectories_by_status(ik_solutions.begin(), ik_solutions.end());
    std::cout << trajectories.size() << "\n";

    return EXIT_SUCCESS;
}

