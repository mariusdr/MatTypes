#ifndef LIBCUMANIP_KINEMATICS_JACOBIAN_HPP
#define LIBCUMANIP_KINEMATICS_JACOBIAN_HPP

#include "../types.hpp"
#include "forward_kinematics.hpp"


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace cumanip 
{

class Jacobian 
{
public:
    
    __device__ __host__
    mt::Matrix<6, 6> solve(const mt::Matrix4f& T1, const mt::Matrix4f T2, const mt::Matrix4f T3,
                           const mt::Matrix4f& T4, const mt::Matrix4f T5, const mt::Matrix4f T6)
    {
        mt::Matrix<6, 6> jacobian;

        mt::Matrix4f T_ee = T6;
        mt::Vector3f p_ee = mt::translation(T_ee);

        mt::Vector3f p0 = mt::vec3f(0.f, 0.f, 0.f);
        mt::Vector3f p1 = mt::translation(T1);
        mt::Vector3f p2 = mt::translation(T2);
        mt::Vector3f p3 = mt::translation(T3);
        mt::Vector3f p4 = mt::translation(T4);
        mt::Vector3f p5 = mt::translation(T5);

        mt::Vector3f z0 = mt::vec3f(0.f, 0.f, 1.f);
        mt::Vector3f z1 = mt::rotation(T1) * z0;
        mt::Vector3f z2 = mt::rotation(T2) * z0;
        mt::Vector3f z3 = mt::rotation(T3) * z0;
        mt::Vector3f z4 = mt::rotation(T4) * z0;
        mt::Vector3f z5 = mt::rotation(T5) * z0;

        mt::Vector3f Jpos1 = mt::cross(z0, (p_ee - p0));
        mt::Vector3f Jpos2 = mt::cross(z1, (p_ee - p1));
        mt::Vector3f Jpos3 = mt::cross(z2, (p_ee - p2));
        mt::Vector3f Jpos4 = mt::cross(z3, (p_ee - p3));
        mt::Vector3f Jpos5 = mt::cross(z4, (p_ee - p4));
        mt::Vector3f Jpos6 = mt::cross(z5, (p_ee - p5));

        mt::Vector3f Jo1 = z0;
        mt::Vector3f Jo2 = z1;
        mt::Vector3f Jo3 = z2;
        mt::Vector3f Jo4 = z3;
        mt::Vector3f Jo5 = z4;
        mt::Vector3f Jo6 = z5;

        jacobian.data[0 * 6 + 0] = Jpos1.data[0];
        jacobian.data[1 * 6 + 0] = Jpos1.data[1];
        jacobian.data[2 * 6 + 0] = Jpos1.data[2];
        jacobian.data[3 * 6 + 0] =   Jo1.data[0];
        jacobian.data[4 * 6 + 0] =   Jo1.data[1];
        jacobian.data[5 * 6 + 0] =   Jo1.data[2];

        jacobian.data[0 * 6 + 1] = Jpos2.data[0];
        jacobian.data[1 * 6 + 1] = Jpos2.data[1];
        jacobian.data[2 * 6 + 1] = Jpos2.data[2];
        jacobian.data[3 * 6 + 1] =   Jo2.data[0];
        jacobian.data[4 * 6 + 1] =   Jo2.data[1];
        jacobian.data[5 * 6 + 1] =   Jo2.data[2];

        jacobian.data[0 * 6 + 2] = Jpos3.data[0];
        jacobian.data[1 * 6 + 2] = Jpos3.data[1];
        jacobian.data[2 * 6 + 2] = Jpos3.data[2];
        jacobian.data[3 * 6 + 2] =   Jo3.data[0];
        jacobian.data[4 * 6 + 2] =   Jo3.data[1];
        jacobian.data[5 * 6 + 2] =   Jo3.data[2];

        jacobian.data[0 * 6 + 3] = Jpos4.data[0];
        jacobian.data[1 * 6 + 3] = Jpos4.data[1];
        jacobian.data[2 * 6 + 3] = Jpos4.data[2];
        jacobian.data[3 * 6 + 3] =   Jo4.data[0];
        jacobian.data[4 * 6 + 3] =   Jo4.data[1];
        jacobian.data[5 * 6 + 3] =   Jo4.data[2];

        jacobian.data[0 * 6 + 4] = Jpos5.data[0];
        jacobian.data[1 * 6 + 4] = Jpos5.data[1];
        jacobian.data[2 * 6 + 4] = Jpos5.data[2];
        jacobian.data[3 * 6 + 4] =   Jo5.data[0];
        jacobian.data[4 * 6 + 4] =   Jo5.data[1];
        jacobian.data[5 * 6 + 4] =   Jo5.data[2];

        jacobian.data[0 * 6 + 5] = Jpos6.data[0];
        jacobian.data[1 * 6 + 5] = Jpos6.data[1];
        jacobian.data[2 * 6 + 5] = Jpos6.data[2];
        jacobian.data[3 * 6 + 5] =   Jo6.data[0];
        jacobian.data[4 * 6 + 5] =   Jo6.data[1];
        jacobian.data[5 * 6 + 5] =   Jo6.data[2];

        return jacobian;
    }
    
    __device__ __host__  
    mt::Matrix<6, 6> solve(const mt::State& state)
    {
        mt::Matrix4f T1, T2, T3, T4, T5, T6;
        
        URKForwardKinematics fk;
        fk.solve(state);
        fk.solutions(T1, T2, T3, T4, T5, T6);

        return solve(T1, T2, T3, T4, T5, T6);
    }
};


}
#endif