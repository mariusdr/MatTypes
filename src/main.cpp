#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <algorithm>
#include <vector>
#include <numeric>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif


#include "planner_cpu.hpp"


namespace cumanip
{


std::vector<mt::Point> read_cartesian_path(const std::string& path)
{
    std::vector<mt::Point> cart;

    std::ifstream is(path);
    if (is)
    {
        float a, b, c, x, y, z;

        while (is >> a >> b >> c >> x >> y >> z)
        {
            mt::Point p = mt::vec6f(x, y, z, a, b, c);
            cart.push_back(p);
        }
    }
    return cart;
}

} // ns






int main(int argc, char** argv)
{
    using namespace cumanip;
    std::cout << std::fixed << std::setprecision(4);

    if (argc < 2)
    {
        std::cerr << "No input file provided\n";
        return EXIT_FAILURE;
    }

    std::string fp(argv[1]);
    std::cout << "reading " << fp << "\n";
    std::vector<mt::Point> cart_path = read_cartesian_path(fp); 

    //cart_path.resize(4);

    std::cout << "read " << cart_path.size() << " points\n";

    PlannerCpu planner(5.f);
    planner.run_in_workspace(cart_path);


}




// int main(int argc, char** argv)
// {
//     using namespace cumanip;
//     std::cout << std::fixed << std::setprecision(4);

//     if (argc < 2)
//     {
//         std::cerr << "No input file provided\n";
//         return EXIT_FAILURE;
//     }

//     std::string fp(argv[1]);
//     std::cout << "reading " << fp << "\n";
//     std::vector<mt::Point> cart_path = read_cartesian_path(fp); 
//     std::vector<mt::Matrix<8, 6>> states(cart_path.size());

//     InverseKinematics<1> ik_op;
//     std::transform(cart_path.begin(), cart_path.end(), states.begin(), ik_op);

//     ComputeTransManip<8> mn_op;
//     std::vector<mt::Vector<8>> manips(cart_path.size());
//     std::transform(states.begin(), states.end(), manips.begin(), mn_op);

//     ComputeDistanceMatrix<8> dm_op;
//     std::vector<mt::Matrix<8, 8>> dist_mats;
//     {
//         for (size_t i = 0; i < states.size() - 1; ++i)
//         {
//             auto fst = states.at(i);
//             auto snd = states.at(i+1);

//             mt::Matrix<8, 8> mat = dm_op(fst, snd);
//             dist_mats.push_back(mat);
//         }
//     }

//     {
//         std::vector<float> tmp(dist_mats.size());

//         MinMat<8, 8> min_op;
//         std::transform(dist_mats.begin(), dist_mats.end(), tmp.begin(), min_op);
//         float min_dist = *std::min_element(tmp.begin(), tmp.end());

//         MaxMat<8, 8> max_op;
//         std::transform(dist_mats.begin(), dist_mats.end(), tmp.begin(), max_op);
//         float max_dist = *std::max_element(tmp.begin(), tmp.end());
        
//         SumMat<8, 8> sum_op;
//         std::transform(dist_mats.begin(), dist_mats.end(), tmp.begin(), sum_op);
//         float avg_dist = std::accumulate(tmp.begin(), tmp.end(), 0.f) / (64 * std::distance(dist_mats.begin(), dist_mats.end()));
        
//         std::cout << "min dist: " << min_dist << "\n";
//         std::cout << "max dist: " << max_dist << "\n";
//         std::cout << "avg dist: " << avg_dist << "\n";
//     }

//     std::vector<mt::Matrix<8, 8>> tms(dist_mats.size());
//     ComputeTransitionMatrix<8> tm_op(4.f);
//     std::transform(dist_mats.begin(), dist_mats.end(), tms.begin(), tm_op);

//     std::vector<mt::Matrix<8, 8>> scanned(dist_mats.size());
//     CombineTransitionMatricies<8> comb_op;
//     scanned[0] = tms[0];
//     for (size_t i = 1; i < dist_mats.size(); ++i)
//     {
//         scanned[i] = comb_op(scanned[i-1], tms[i]);
//     }
    
//     for (auto it = scanned.begin(); it != scanned.end(); ++it)
//     {
//         std::cout << *it << "\n\n";
//     }

//     return EXIT_SUCCESS;
// }

