#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Types.hpp"
#include "kin.h"

void check_cleaned_fwd()
{
    std::srand(std::time(nullptr));

    for (int i = 0; i < 1000000; ++i)
    {
        mt::State s;
        for (int j = 0; j < 6; ++j)
        {
            s.data[j] = float(std::rand() % 100 + 1.f) / 10.f;
        }

        mt::Matrix4f sol1 = ur_kin::solveFK(s);
        mt::Matrix4f sol2 = ur_kin::solveFK_(s);

        if (!sol1.approx_equal(sol2))
        {
            std::cout << "error in state s = " << s << " ! \n";
            std::cout << "sol1:\n" << sol1 << "sol2:\n" << sol2 << "\n";
        }
    }
}

 
int main()
{
    auto s = mt::state(0.12, 0.34, M_PI, M_PI_2, 0.21, 0.11);

    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    std::cout << "ref:\n" << ur_kin::solveFK(s) << "\n";

    auto t01 = ur_kin::get_dh_transform(0, s);
    auto t12 = ur_kin::get_dh_transform(1, s);
    auto t23 = ur_kin::get_dh_transform(2, s);
    auto t34 = ur_kin::get_dh_transform(3, s);
    auto t45 = ur_kin::get_dh_transform(4, s);
    auto t56 = ur_kin::get_dh_transform(5, s);

    // std::cout << "t01:\n" << t01 << "\n";
    // std::cout << "t12:\n" << t12 << "\n";
    // std::cout << "t23:\n" << t23 << "\n";
    // std::cout << "t34:\n" << t34 << "\n";
    // std::cout << "t45:\n" << t45 << "\n";
    // std::cout << "t56:\n" << t56 << "\n";


    auto tfull = t56 * t45 * t34 * t23 * t12 * t01;
    
    std::cout << "tfull:\n" << tfull << "\n";

    return EXIT_SUCCESS;
}