#ifndef LIBCUMANIP_MANIPULABILITY_OPS_H
#define LIBCUMANIP_MANIPULABILITY_OPS_H

#include <cstring>

//!
#include "../math_types.hpp"
#include "../kinematics.hpp"
//!

namespace cumanip
{


template <size_t NumStates>
struct ComputeTransManip
{
    __host__ __device__
    mt::Vector<NumStates> operator()(const mt::Matrix<NumStates, 6>& states)
    {
        mt::Vector<NumStates> v(-1.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            cumanip::Manipulability algo;
            algo.solve(states.get_row(i));
            v.at(i) = algo.trans_manip();
        }
        return v;
    }
};

template <>
struct ComputeTransManip<1>
{
    __host__ __device__
    float operator()(const mt::State& state)
    {
        cumanip::Manipulability algo;
        algo.solve(state);
        return algo.trans_manip();
    }
};

template <size_t NumStates>
struct ComputeRotManip
{
    __host__ __device__
    mt::Vector<NumStates> operator()(const mt::Matrix<NumStates, 6>& states)
    {
        mt::Vector<NumStates> v(-1.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            cumanip::Manipulability algo;
            algo.solve(states.get_row(i));
            v.at(i) = algo.rot_manip();
        }
        return v;
    }
};

template <>
struct ComputeRotManip<1>
{
    __host__ __device__
    float operator()(const mt::State& state)
    {
        cumanip::Manipulability algo;
        algo.solve(state);
        return algo.rot_manip();
    }
};

template <size_t NumStates>
struct ComputeFullManip
{
    __host__ __device__
    mt::Vector<NumStates> operator()(const mt::Matrix<NumStates, 6>& states)
    {
        mt::Vector<NumStates> v(-1.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            cumanip::FullManipulability algo;
            algo.solve(states.get_row(i));
            v.at(i) = algo.manip();
        }
        return v;
    }
};

template <>
struct ComputeFullManip<1>
{
    __host__ __device__
    float operator()(const mt::State& state)
    {
        cumanip::FullManipulability algo;
        algo.solve(state);
        return algo.manip();
    }
};



} // namespace
#endif 