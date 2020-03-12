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
// #include "kinematics.hpp"
// #include "forward_kinematics_op.h"
// #include "inverse_kinematics_op.h"
// #include "common_op.h"
// #include "extract_traj_by_status.h"

#include "math_types/svd.hpp"

#include <vector>




int main()
{
    using namespace cumanip;

    std::cout << std::fixed << std::setprecision(4);

    mt::Matrix<9, 7> mat 
    {
    0.0495,   0.2819,   0.8739,   0.1057,   0.0634,   0.1339,   0.8422,
    0.4896,   0.5386,   0.2703,   0.1420,   0.8604,   0.0309,   0.5590,
    0.1925,   0.6952,   0.2085,   0.1665,   0.9344,   0.9391,   0.8541,
    0.1231,   0.4991,   0.5650,   0.6210,   0.9844,   0.3013,   0.3479,
    0.2055,   0.5358,   0.6403,   0.5737,   0.8589,   0.2955,   0.4460,
    0.1465,   0.4452,   0.4170,   0.0521,   0.7856,   0.3329,   0.0542,
    0.1891,   0.1239,   0.2060,   0.9312,   0.5134,   0.4671,   0.1771,
    0.0427,   0.4904,   0.9479,   0.7287,   0.1776,   0.6482,   0.6628,
    0.6352,   0.8530,   0.0821,   0.7378,   0.3986,   0.0252,   0.3308
    };


    mt::SVD<9, 7> svd(mat);    
    svd.solve();

    std::cout << svd.get_inp() << "\n\n\n";
    
    // std::cout << svd.get_u() << "\n\n\n";
    std::cout << svd.get_s() << "\n\n\n";
    // std::cout << svd.get_v() << "\n\n\n";

    std::cout << svd.get_s().drop_last_row().drop_last_row() << "\n\n";


    return EXIT_SUCCESS;
}

