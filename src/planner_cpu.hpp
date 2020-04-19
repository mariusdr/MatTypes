#ifndef LIBCUMANIP_MOTION_PLANNER_CPU_HPP
#define LIBCUMANIP_MOTION_PLANNER_CPU_HPP

//!
#include "math_types.hpp"
#include "typedefs.h"
//!

#include <vector>

namespace cumanip
{


class PlannerCpu
{
public:

    PlannerCpu(mt::Matrix<6, 6> distance_weights, float max_dist, mt::Matrix4f offset_transformation): 
    distance_weights(distance_weights), max_dist(max_dist), offset_transformation(offset_transformation)
    {}
    
    PlannerCpu(mt::Matrix<6, 6> distance_weights, float max_dist): 
    distance_weights(distance_weights), max_dist(max_dist), offset_transformation(mt::identity<4, 4>())
    {}

    PlannerCpu(float max_dist): 
    distance_weights(mt::identity<6, 6>()), max_dist(max_dist), offset_transformation(mt::identity<4, 4>())
    {}

    PlannerCpu():
    distance_weights(mt::identity<6, 6>()), max_dist(1.f), offset_transformation(mt::identity<4, 4>())
    {}

    void run_in_workspace(std::vector<mt::Point>& inp);

    void get_manip_of_traj(const std::vector<mt::State>& inp);

    void verify(const std::vector<mt::State>& states, const std::vector<mt::Point>& points);

private:
    
    struct PathTrace
    {
        std::vector<size_t> indices;
        float manip;
        
        size_t tail_i;
        size_t tail_j;
    };

    mt::Matrix<6, 6> distance_weights;
    float max_dist;
    
    mt::Matrix4f offset_transformation;

    void compute_ik_solutions(const std::vector<mt::Point>& points); 
    void compute_manip_matricies();
    
    std::vector<PlannerCpu::PathTrace> backtrace(const std::vector<mt::Point>& points);


    std::vector<mt::Matrix<8, 6>> ik_solutions;
    std::vector<mt::Matrix<8, 8>> transition_matricies;
    std::vector<mt::Matrix<8, 8>> manip_matricies;
    std::vector<mt::Matrix<8, 8>> scanned_manip_matricies;
    std::vector<mt::Matrix<8, 8>> scanned_manip_matricies_transitions;

};





} // namespace

#endif