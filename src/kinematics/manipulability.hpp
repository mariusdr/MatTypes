#ifndef LIBCUMANIP_KINEMATICS_MANIPULABILITY_HPP
#define LIBCUMANIP_KINEMATICS_MANIPULABILITY_HPP

#include "../types.hpp"
#include "jacobian.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace cumanip 
{

class Manipulability 
{
public:
    
    __device__ __host__
    Manipulability(): translation_manipulability(-1.f), rotation_manipulability(-1.f) 
    {}

    __device__ __host__ 
    void solve(const mt::State& state)
    {
        Jacobian algo;
        mt::Matrix<6, 6> jacobian = algo.solve(state);
        
        // split jacobian into translation and rotation parts
        mt::Matrix<3, 6> translation_jacobian;
        mt::Matrix<3, 6> rotation_jacobian;

        translation_jacobian.set_row(0, jacobian.get_row(0));
        translation_jacobian.set_row(1, jacobian.get_row(1));
        translation_jacobian.set_row(2, jacobian.get_row(2));

        rotation_jacobian.set_row(0, jacobian.get_row(3));
        rotation_jacobian.set_row(1, jacobian.get_row(4));
        rotation_jacobian.set_row(2, jacobian.get_row(5));

        // compute their determinants 
        mt::Matrix<3, 3> t = translation_jacobian * translation_jacobian.transpose();
        mt::Matrix<3, 3> r = rotation_jacobian * rotation_jacobian.transpose(); 

        translation_manipulability = sqrt(mt::determinant(t));
        rotation_manipulability = sqrt(mt::determinant(r));
    }

    __device__ __host__ 
    float rot_manip() const 
    {
        return rotation_manipulability;
    }

    __device__ __host__ 
    float trans_manip() const 
    {
        return translation_manipulability;
    }

private:
    float translation_manipulability;
    float rotation_manipulability;
};


class FullManipulability 
{
public:

    __device__ __host__
    FullManipulability(): manipulability(-1.f)
    {}

    __device__ __host__ 
    void solve(const mt::State& state)
    {
        Jacobian algo;
        mt::Matrix<6, 6> jacobian = algo.solve(state);
        mt::Matrix<6, 6> t = jacobian * jacobian.transpose();
        manipulability = sqrt(mt::determinant(t));
    }

    __device__ __host__
    float manip() const 
    {
        return manipulability;
    }

private:
    float manipulability;
};




} // namespace
#endif