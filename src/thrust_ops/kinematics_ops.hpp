#ifndef LIBCUMANIP_KINEMATICS_OPS_H
#define LIBCUMANIP_KINEMATICS_OPS_H

#include <cstring>

//!
#include "../math_types.hpp"
#include "../kinematics.hpp"
//!

namespace cumanip
{

struct ForwardKinematics
{
    __host__ __device__ 
    mt::Matrix4f operator()(const mt::State& state)
    {
        URKForwardKinematics fk;
        fk.solve(state);
        return fk.transform_base_to_ee(); 
    }
};

template <size_t NumPoints>
struct InverseKinematics
{
    static const size_t NumSolutions = 8 * NumPoints;

    __host__ __device__
    mt::Matrix<NumSolutions, 6> operator()(const mt::Matrix<NumPoints, 6>& poses)
    {
        mt::Matrix<NumSolutions, 6> states;
        URKInverseKinematics ik;
        
        for (size_t i = 0; i < NumPoints; ++i)
        {
            mt::Point pt = poses.get_row(i);
            ik.solve(pt);
            const mt::Matrix<8, 6>& sol = ik.get_solutions();

            states.set_row(8 * i + 0, sol.get_row(0));
            states.set_row(8 * i + 1, sol.get_row(1));
            states.set_row(8 * i + 2, sol.get_row(2));
            states.set_row(8 * i + 3, sol.get_row(3));
            states.set_row(8 * i + 4, sol.get_row(4));
            states.set_row(8 * i + 5, sol.get_row(5));
            states.set_row(8 * i + 6, sol.get_row(6));
            states.set_row(8 * i + 7, sol.get_row(7));
        }
        return states;
    }
};

template <>
struct InverseKinematics<1>
{
    __host__ __device__
    mt::Matrix<8, 6> operator()(const mt::Point& pose)
    {
        URKInverseKinematics ik;
        ik.solve(pose);
        return ik.get_solutions();    
    }

    __host__ __device__
    mt::Matrix<8, 6> operator()(const mt::Matrix4f& affine)
    {
        URKInverseKinematics ik;
        ik.solve(affine);
        return ik.get_solutions();    
    }
};




template <size_t NumSamples>
struct RunTranslation3DIkOp
{
    static const size_t NumRows = NumSamples * 8;

    using OutMatrix = mt::Matrix<NumRows, 6>;
    using InMatrix = mt::Matrix<NumSamples, 3>;

    __device__ __host__
    OutMatrix operator()(const mt::Vector3f& xyz, const InMatrix& rpy_vals)
    {
        OutMatrix states(0.f);

        for (size_t i = 0; i < NumSamples; ++i)
        {
            mt::Vector3f rpy = rpy_vals.get_row(i);
            mt::Matrix4f pose = mt::affine(mt::fromRPY(rpy), xyz);

            URKInverseKinematics ik;
            size_t n = ik.solve(pose);
            const mt::Matrix<8, 6>& sol = ik.get_solutions();

            states.set_row(8 * i + 0, sol.get_row(0));
            states.set_row(8 * i + 1, sol.get_row(1));
            states.set_row(8 * i + 2, sol.get_row(2));
            states.set_row(8 * i + 3, sol.get_row(3));
            states.set_row(8 * i + 4, sol.get_row(4));
            states.set_row(8 * i + 5, sol.get_row(5));
            states.set_row(8 * i + 6, sol.get_row(6));
            states.set_row(8 * i + 7, sol.get_row(7));
        }

        return states;
    }

    // __device__ __host__
    // OutMatrix operator()(const thrust::tuple<mt::Vector3f, InMatrix>& p)
    // {
    //     return (*this)(thrust::get<0>(p), thrust::get<1>(p));
    // }

};












} // namespace
#endif