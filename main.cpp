#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Types.hpp"
#include "kin.h"

 
int main()
{
    mt::State s = mt::state(0.111664,-0.420859,0.000141,0.229311,0.033719,-0.158429);

    mt::Matrix<8,6> sol(0.f);
    unsigned char status[8];
    int nums = ur_kin::backward(ur_kin::solveFK(s).data, sol.data, status, 0.f, ur_kin::UR_5_DH);

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "sol: " << sol << "\n";
    std::cout << "num of solutions: " << nums << "\n";

    return EXIT_SUCCESS;
}