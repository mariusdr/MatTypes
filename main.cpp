#include <iostream>
#include <iomanip>

#include "Types.hpp"
#include "kin.h"

int main()
{
    auto s = mt::state(0.12, 0.34, M_PI, M_PI_2, 0.21, 0.11);

    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    std::cout << ur_kin::solveFK(s) << "\n";
    std::cout << ur_kin::solveFK_(s) << "\n";

    return EXIT_SUCCESS;
}