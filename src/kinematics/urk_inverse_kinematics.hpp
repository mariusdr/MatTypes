#ifndef LIBCUMANIP_KINEMATICS_URK_INVERSE_KINEMATICS_HPP
#define LIBCUMANIP_KINEMATICS_URK_INVERSE_KINEMATICS_HPP

#include "cuda_ur_kinematics.hpp"
#include "../types.hpp"

namespace cumanip
{


class URKInverseKinematics
{
public:

    __device__ __host__ 
    int solve(const mt::Matrix4f& T, mt::Matrix<8, 6>& solutions, unsigned char* status, float q6_des)
    {
        solutions = mt::m_zeros<8, 6>(); 
        int num_sols = ur_kin::backward(T.data, solutions.data, status, q6_des);
        return num_sols;
    }

    __device__ __host__ 
    void statusToStr(unsigned char status, char* str)
    {
        unsigned char i = status & 0x04;
        unsigned char j = status & 0x02;
        unsigned char k = status & 0x01;

        const char* shoulder = (i == 0) ? "left" : "right";
        const char* wrist = (j == 0) ? "up" : "down";
        const char* elbow = (k == 0) ? "up" : "down";

        sprintf(str, "shoulder %s, wrist %s, elbow %s\n", shoulder, wrist, elbow);
    }

private:
        
};





} // namespace
#endif