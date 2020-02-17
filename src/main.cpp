#include <iostream>
#include <iomanip>
#include <cstdlib>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


#include "Types.hpp"
#include "kin.h"


int main()
{
    mt::State s = mt::state(0.12, 1.23, 0.23, 0.44, -0.22, 1.31);

    auto J = ur_kin::compute_jacobian(s);

    std::cout << J << "\n";

    return EXIT_SUCCESS;
}