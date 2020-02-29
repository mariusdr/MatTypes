#ifndef LIBCUMANIP_KINEMATICS_FORWARD_KINEMATICS_HPP
#define LIBCUMANIP_KINEMATICS_FORWARD_KINEMATICS_HPP

#include "cuda_ur_kinematics.hpp"
#include "../types.hpp"

namespace cumanip
{

class URKForwardKinematics
{
public:

    __device__ __host__
    URKForwardKinematics(): t1(mt::identity<4, 4>()), t2(mt::identity<4, 4>()), t3(mt::identity<4, 4>()),
                            t4(mt::identity<4, 4>()), t5(mt::identity<4, 4>()), t6(mt::identity<4, 4>()),
                            base_transform(mt::identity<4, 4>()), ee_transform(mt::identity<4, 4>())
    {
        // these are the defaults for the RPS
        base_transform.set_row(0, mt::vec4f(-1, 0, 0, 0));
        base_transform.set_row(1, mt::vec4f(0, -1, 0, 0));
        base_transform.set_row(2, mt::vec4f(0, 0, 1, 0));
        base_transform.set_row(3, mt::vec4f(0, 0, 0, 1));

        ee_transform.set_row(0, mt::vec4f(0, -1, 0, 0));
        ee_transform.set_row(1, mt::vec4f(0, 0, -1, 0));
        ee_transform.set_row(2, mt::vec4f(1, 0, 0, 0));
        ee_transform.set_row(3, mt::vec4f(0, 0, 0, 1));
    }

    __device__ __host__
    URKForwardKinematics(mt::Matrix4f base_transform, mt::Matrix4f ee_transform): 
        t1(mt::identity<4, 4>()), t2(mt::identity<4, 4>()), t3(mt::identity<4, 4>()),
        t4(mt::identity<4, 4>()), t5(mt::identity<4, 4>()), t6(mt::identity<4, 4>()),
        base_transform(base_transform), ee_transform(ee_transform)
    {
    }

    __device__ __host__ 
    void solve(const mt::State& state)
    {
        ur_kin::forward_all(state.data, t1.data, t2.data, t3.data, t4.data, t5.data, t6.data);
        t1 = base_transform * t1;
        t2 = base_transform * t2;
        t3 = base_transform * t3;
        t4 = base_transform * t4;
        t5 = base_transform * t5;
        t6 = base_transform * t6 * ee_transform;
    }

    __device__ __host__ 
    void solutions(mt::Matrix4f& T1, mt::Matrix4f& T2, mt::Matrix4f& T3, 
                   mt::Matrix4f& T4, mt::Matrix4f& T5, mt::Matrix4f& T6)
    {
        T1 = t1;
        T2 = t2;
        T3 = t3;
        T4 = t4;
        T5 = t5;
        T6 = t6;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j1() const 
    {
        return t1;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j2() const 
    {
        return t2;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j3() const 
    {
        return t3;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j4() const 
    {
        return t4;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j5() const 
    {
        return t5;
    }

    __device__ __host__ 
    mt::Matrix4f transform_base_to_j6() const 
    {
        mt::Matrix4f inv_ee_transform;
        mt::invert(ee_transform, inv_ee_transform);
        return t6 * inv_ee_transform;
    }
    
    __device__ __host__ 
    mt::Matrix4f transform_base_to_ee() const 
    {
        return t6;
    }

    __device__ __host__ 
    mt::Matrix4f get_base_transform() const 
    {
        return base_transform;
    }

    __device__ __host__ 
    mt::Matrix4f get_ee_transform() const 
    {
        return ee_transform;
    }

private:

    // transforms from the zeroth (!) to the i-th link
    mt::Matrix4f t1;
    mt::Matrix4f t2;
    mt::Matrix4f t3;
    mt::Matrix4f t4;
    mt::Matrix4f t5;
    mt::Matrix4f t6;

    // transform from base to zeroth link
    mt::Matrix4f base_transform; 

    // transform from sixth linkt to end effector
    mt::Matrix4f ee_transform;
};

} // namespace
#endif