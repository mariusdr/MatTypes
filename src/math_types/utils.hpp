#ifndef LIBCUMANIP_TYPES_UTILS_HPP
#define LIBCUMANIP_TYPES_UTILS_HPP

#include "vector.hpp"

namespace cumanip 
{
namespace mt 
{

template <size_t Len>
__host__ __device__ inline
void sort_vector_inp(mt::Vector<Len>& vec)
{
    for (int i = 1; i < Len; ++i)
    {
        float x = vec.at(i);
        int j = i - 1;

        while (j >= 0 && vec.at(j) < x)
        {
            vec.at(j + 1) = vec.at(j);
            j = j - 1;
        }
        vec.at(j+1) = x;
    }
}



} // ns
} // ns
#endif