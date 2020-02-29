#include <iostream>
#include <iomanip>
#include <cstdlib>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


#include "types.hpp"
#include "kinematics.hpp"

int main()
{
    cumanip::mt::State s = cumanip::mt::state(0.12, 1.23, 0.23, 0.44, -0.22, 1.31);

    // cumanip::Manipulability algo;
    // algo.solve(s);
    // std::cout << algo.trans_manip() << "\n";
    // std::cout << algo.rot_manip() << "\n";

    // cumanip::URKForwardKinematics fk;
    // fk.solve(s);
    // std::cout << fk.transform_base_to_ee() << "\n";

    return EXIT_SUCCESS;
}

