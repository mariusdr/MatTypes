//!
#include "planner_cpu.hpp"
#include "thrust_ops.hpp"
#include "typedefs.h"
//!

#include <algorithm>
#include <cstddef>
#include <functional>


namespace cumanip 
{

void PlannerCpu::compute_ik_solutions(const std::vector<mt::Point>& points)
{
    ik_solutions.clear();
    ik_solutions = std::vector<mt::Matrix<8, 6>>(points.size());

    InverseKinematics<1> ik_op;
    std::transform(points.begin(), points.end(), ik_solutions.begin(), ik_op);
}


void PlannerCpu::compute_manip_matricies()
{
    size_t n = ik_solutions.size();

    transition_matricies.clear();
    manip_matricies.clear();
    scanned_manip_matricies.clear();

    // transition_matricies = std::vector<mt::Matrix<8, 8>>(n - 1);
    // manip_matricies = std::vector<mt::Matrix<8, 8>>(n - 1);

    ComputeDistanceMatrix<8> dop(distance_weights);
    ComputeTransitionMatrix<8> top(max_dist);
    ComputeTransManip<8> manip_op;
    ComputeManipMatricies<8> mm_op;

    for (size_t i = 0; i < n - 1; ++i)
    {
        auto sol_from = ik_solutions.at(i);
        auto sol_to = ik_solutions.at(i + 1);
        mt::Matrix<8, 8> tm = top(dop(sol_from, sol_to));
        
        mt::Vector<8> manip_from = manip_op(sol_from);
        mt::Vector<8> manip_to = manip_op(sol_to);
        manip_matricies.push_back(mm_op(manip_from, tm, manip_to)); 
    }

    // scan  
    
    scanned_manip_matricies = std::vector<mt::Matrix<8, 8>>(n - 1);
    scanned_manip_matricies_transitions = std::vector<mt::Matrix<8, 8>>(n - 1);

    CombineManipMatricies<8> comb_op;
    CombineManipMatriciesTransitions<8> trans_op;

    scanned_manip_matricies_transitions.at(0) = mt::Matrix<8, 8>(-1.f);
    scanned_manip_matricies.at(0) = manip_matricies.at(0);
    for (size_t i = 1; i < n - 1; ++i)
    { 
        mt::Matrix<8, 8> fst = scanned_manip_matricies.at(i - 1);
        mt::Matrix<8, 8> snd = manip_matricies.at(i);
        scanned_manip_matricies.at(i) = comb_op(fst, snd);

        scanned_manip_matricies_transitions.at(i) = trans_op(fst, snd);
    }


}

void PlannerCpu::run_in_workspace(std::vector<mt::Point>& points)
{
    compute_ik_solutions(points);
    std::cout << "computed " << ik_solutions.size() << " ik solutions \n\n";

    compute_manip_matricies();
    std::cout << "computed " << manip_matricies.size() << " manip matricies...\n";

    backtrace(points);
}


std::vector<PlannerCpu::PathTrace> PlannerCpu::backtrace(const std::vector<mt::Point>& points)
{
    std::vector<PathTrace> paths;

    std::function<void(size_t, size_t, size_t, PathTrace)> trace = [&](size_t cur_mat, size_t cur_i, size_t cur_j, PathTrace path)
    {
        path.indices.push_back(cur_j);

        if (cur_mat == 0)
        {
            path.indices.push_back(cur_i);
            paths.push_back(path);
            return;
        }

        mt::Matrix8f tm = scanned_manip_matricies.at(cur_mat);
        mt::Vector<8> column = tm.get_col(cur_j);

        mt::Matrix8f tm_inds = scanned_manip_matricies_transitions.at(cur_mat);
        mt::Vector<8> column_inds = tm_inds.get_col(cur_j); 

        for (int k = 0; k < 8; ++k)
        {
            int k_idx = size_t(column_inds.at(k));
            if (k_idx > -1)
            {
                PathTrace copy;
                copy.indices = path.indices;
                copy.manip = path.manip;
                copy.tail_i = path.tail_i;
                copy.tail_j = path.tail_j;
                trace(cur_mat - 1, cur_i, k_idx, copy);
                break;
            }
        }
    };

    size_t n = scanned_manip_matricies.size();
    mt::Matrix8f tail = scanned_manip_matricies.at(n - 1);

    PathTrace p;
    p.tail_i = 5;
    p.tail_j = 6;
    p.manip = tail.at(5, 6);
    trace(n - 1, 5, 6, p);

    // for (size_t i = 0; i < 8; ++i)
    // {
    //     for (size_t j = 0; j < 8; ++j)
    //     {
    //         if (tail.at(i, j) > 0.f)
    //         {
    //             PathTrace p;
    //             p.tail_i = i;
    //             p.tail_j = j;
    //             p.manip = tail.at(i, j);
    //             trace(n - 1, i, j, p);
    //         }
    //     }
    // }
    
    // size_t n = scanned_manip_matricies.size();
    // mt::Matrix8f tail = scanned_manip_matricies.at(n - 1);
    // size_t i, j;
    // float m = tail.max(i, j);
    // if (m > 0.f)
    // {
    //     PathTrace p;
    //     p.manip = tail.at(i, j);
    //     trace(n - 1, i, j, p);
    // }

    std::cout << "found " << paths.size() << " path traces\n\n";

    std::function<void(const std::vector<size_t>&, std::vector<mt::State>&)> collect_states 
    = [&](const std::vector<size_t>& inds, std::vector<mt::State>& states)
    {
        auto ind_it = inds.rbegin();
        for (auto it = ik_solutions.begin(); it != ik_solutions.end(); ++it)
        {
            const mt::Matrix<8, 6>& sol = *it;
            const size_t idx = *ind_it;
            ind_it++;
            states.push_back(sol.get_row(idx));
        }
    };

    for (PathTrace p: paths)
    {

        std::cout << "analyze path trace (" << p.tail_i << ", " << p.tail_j << ") with manip " << p.manip << " ...\n";
        std::cout << "path trace has " << p.indices.size() << " indices \n";

        std::vector<mt::State> states;
        collect_states(p.indices, states);

        verify(states, points);

        get_manip_of_traj(states);

        std::cout << "=======================================\n\n";

    }

    return paths;
}

void PlannerCpu::get_manip_of_traj(const std::vector<mt::State>& states)
{
    std::cout << "analyze traj with " << states.size() << " states\n";    
    
    std::vector<float> mvs(states.size());
    std::transform(states.begin(), states.end(), mvs.begin(), ComputeTransManip<1>());

    float min_val = *std::min_element(mvs.begin(), mvs.end());
    float max_val = *std::max_element(mvs.begin(), mvs.end());

    std::cout << "min: " << min_val << " max: " << max_val << "\n";
}

void PlannerCpu::verify(const std::vector<mt::State>& states, const std::vector<mt::Point>& points)
{
    std::vector<mt::Point> res(points.size());
    std::transform(states.begin(), states.end(), res.begin(), ForwardKinematicsPts());
    
    if (points.size() != res.size())
    {
        std::cout << "reference (" << points.size() << ") and result (" << res.size() << ") sizes are different\n";
    }

    for (int i = 0; i < points.size(); ++i)
    {
        mt::Matrix4f ref_tf = mt::point_to_affine(points.at(i));
        mt::Matrix4f res_tf = mt::point_to_affine(res.at(i));

        bool eq = ref_tf.approx_equal(res_tf, 0.001);

        if (!eq)
        {
            std::cout << "entry " <<  i << " is unequal\n";
            
            std::cout << ref_tf << "\n";
            std::cout << res_tf << "\n";
        }
    }
    std::cout << "result and reference trajectories are equal\n";


}


} // namespace