#ifndef LIBCUMANIP_KINEMATICS_URK_INVERSE_KINEMATICS_HPP
#define LIBCUMANIP_KINEMATICS_URK_INVERSE_KINEMATICS_HPP

#include "cuda_ur_kinematics.hpp"
#include "../math_types.hpp"

#include <cstring>

namespace cumanip
{


class URKInverseKinematics
{
public:

    __device__ __host__
    URKInverseKinematics(): number_of_solutions(0), solutions(mt::Matrix<8, 6>(0.f)) 
    {
        memset(status, 0, sizeof(unsigned char) * 8);
    }

    __device__ __host__ 
    int solve(const mt::Matrix4f& T, float q6_des)
    {
        number_of_solutions = ur_kin::backward(T.data, solutions.data, status, q6_des);
        return number_of_solutions;
    }

    __device__ __host__ 
    int solve(const mt::Matrix4f& T)
    {
        number_of_solutions = ur_kin::backward(T.data, solutions.data, status, 0.f);
        return number_of_solutions;
    }

    __device__ __host__ 
    int solve(const mt::Point& pt, float q6_des)
    {
        return solve(mt::point_to_affine(pt), q6_des);
    }

    __device__ __host__ 
    int solve(const mt::Point& pt)
    {
        return solve(mt::point_to_affine(pt));
    }

    __device__ __host__
    int get_number_of_solutions() const 
    {
        return number_of_solutions;
    }

    __device__ __host__ 
    mt::Matrix<8, 6> get_solutions() const 
    {
        return solutions;
    }

    __device__ __host__
    unsigned char* get_status() 
    {
        return status;
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
    int number_of_solutions;
    mt::Matrix<8, 6> solutions;
    unsigned char status[8];

};





} // namespace
#endif