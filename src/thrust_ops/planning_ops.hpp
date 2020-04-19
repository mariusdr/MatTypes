#ifndef LIBCUMANIP_PLANNING_OPS_H
#define LIBCUMANIP_PLANNING_OPS_H

#include <cstddef>
#include <cstring>

//!
#include "../math_types.hpp"
#include "../kinematics.hpp"
#include "../cuda_utils.h"
//!


namespace cumanip
{

template <size_t NumStates>
struct CollisionDetection
{
    using MatrixN = mt::Matrix<NumStates, 6>;

    __host__ __device__
    MatrixN operator()(const MatrixN& states)
    {
        
    }
};


template <size_t NumStates>
struct ComputeDistanceMatrix /* : public thrust::binary_function<mt::Matrix<NumStates, 6>, mt::Matrix<NumStates, 6>, mt::Matrix<NumStates, NumStates>>*/
{
    using InMatrix = mt::Matrix<NumStates, 6>;
    using OutMatrix = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    ComputeDistanceMatrix(mt::Matrix<6, 6> joint_weights): joint_weights(joint_weights)
    {}
    
    __device__ __host__ 
    ComputeDistanceMatrix(): joint_weights(mt::identity<6, 6>())
    {}

    __device__ __host__ 
    OutMatrix operator()(const InMatrix& fst, const InMatrix& snd)
    {
        OutMatrix dm(infty());

        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                const mt::State& si = fst.get_row(i);
                const mt::State& sj = snd.get_row(j);
                dm.at(i, j) = mt::distance(joint_weights, si, sj);
            }
        }
        return dm;
    }
    
    // __device__ __host__
    // OutMatrix operator()(const thrust::tuple<InMatrix, InMatrix>& p)
    // {
    //     return (*this)(thrust::get<0>(p), thrust::get<1>(p));
    // }

    mt::Matrix<6, 6> joint_weights;
};


template <size_t NumStates>
struct ComputeTransitionMatrix /*: public thrust::unary_function<mt::Matrix<NumStates, NumStates>, mt::Matrix<NumStates, NumStates>>*/
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    ComputeTransitionMatrix(float md): max_dist(md)
    {}

    __device__ __host__
    MatrixN operator()(const MatrixN& distance_matrix)
    {
        MatrixN out(0.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                const float dist = distance_matrix.at(i, j);
                out.at(i, j) = (dist <= max_dist) ? 1.f : 0.f;
            }
        }
        return out;
    }
    
    float max_dist;
};


template <size_t NumStates>
struct ComputeManipMatricies /* : public thrust::unary_function<thrust::tuple<mt::Vector<NumStates>, mt::Matrix<NumStates, NumStates>, mt::Vector<NumStates>>, mt::Matrix<NumStates, NumStates>>*/
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;
    using VectorN = mt::Vector<NumStates>;

    // using VecMatVec = thrust::tuple<VectorN, MatrixN, VectorN>;
    using VecMatVec = std::tuple<VectorN, MatrixN, VectorN>;

    __device__ __host__ 
    MatrixN operator()(const VectorN& manip_from, const MatrixN& tm, const VectorN& manip_to)
    {
        MatrixN out(0.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                const float& mi = manip_from.at(i);
                const float& mj = manip_to.at(j);
                // const float& m = thrust::min(mi, mj);
                const float& m = std::min(mi, mj);
                out.at(i, j) = tm.at(i, j) > 0.f ? m : 0.f;
            }
        }
        return out;
    }

    // __device__ __host__ 
    // MatrixN operator()(const VecMatVec& vmv)
    // {
    //     return (*this)(thrust::get<0>(vmv), thrust::get<1>(vmv), thrust::get<2>(vmv));
    // }
};


template <size_t NumStates>
struct CombineManipMatricies /* : public thrust::binary_function<mt::Matrix<NumStates, NumStates>, mt::Matrix<NumStates, NumStates>, mt::Matrix<NumStates, NumStates>>*/
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    MatrixN operator()(const MatrixN& fst, const MatrixN& snd)
    {
        MatrixN out(0.f);
        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                float v = 0.f;
                for (size_t k = 0; k < NumStates; ++k)
                {
                    float x_ik = fst.at(i, k);
                    float x_kj = snd.at(k, j);

                    if (x_ik > 0.f && x_kj > 0.f)
                    {
                        // float m = thrust::min(x_ik, x_kj);
                        // v = thrust::max(v, m);
                        float m = std::min(x_ik, x_kj);
                        v = std::max(v, m);
                    }
                }
                out.at(i, j) = v;
            }
        }
        return out;
    }
};

template <size_t NumStates>
struct CombineManipMatriciesTransitions /* : public thrust::binary_function<mt::Matrix<NumStates, NumStates>, mt::Matrix<NumStates, NumStates>, mt::Matrix<NumStates, NumStates>>*/
{
    using MatrixN = mt::Matrix<NumStates, NumStates>;

    __device__ __host__ 
    MatrixN operator()(const MatrixN& fst, const MatrixN& snd)
    {
        MatrixN out(-1.f);

        for (size_t i = 0; i < NumStates; ++i)
        {
            for (size_t j = 0; j < NumStates; ++j)
            {
                float v = 0.f;
                int max_k = -1;

                for (size_t k = 0; k < NumStates; ++k)
                {
                    float x_ik = fst.at(i, k);
                    float x_kj = snd.at(k, j);

                    if (x_ik > 0.f && x_kj > 0.f)
                    {
                        // float m = thrust::min(x_ik, x_kj);
                        // v = thrust::max(v, m);
                        float m = std::min(x_ik, x_kj);
                        
                        if (m > v)
                        {
                            max_k = k;
                        }

                        v = std::max(v, m);
                    }
                }
                out.at(i, j) = max_k;
            }
        }
        return out;
    }
};



} // namespace
#endif