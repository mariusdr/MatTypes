#ifndef LIBCUMANIP_TYPEDEFS_H
#define LIBCUMANIP_TYPEDEFS_H

#include "kinematics.hpp"

#include <vector>

namespace cumanip
{

using StateTrajectorySTL = std::vector<mt::State>;
//!
using StateTrajectoryDev = StateTrajectorySTL;
using StateTrajectoryHost = StateTrajectorySTL;
//!

using Point_STLIter = std::vector<mt::Point>::iterator;
//!
using Point_DevIter = Point_STLIter;
using Point_HostIter = Point_STLIter;
//!

using Mat4_STLIter = std::vector<mt::Matrix4f>::iterator;
//!
using Mat4_DevIter = Mat4_STLIter;
using Mat4_HostIter = Mat4_STLIter;
//!

using Mat8_STLIter = std::vector<mt::Matrix8f>::iterator;
//!
using Mat8_DevIter = Mat8_STLIter;
using Mat8_HostIter = Mat8_STLIter;
//!

using Vec8_STLIter = std::vector<mt::Vector<8>>::iterator;
//!
using Vec8_DevIter = Vec8_STLIter;
using Vec8_HostIter = Vec8_STLIter;
//!

using State_STLIter = std::vector<mt::State>::iterator;
//!
using State_DevIter = State_STLIter;
using State_HostIter = State_STLIter;
//!

struct IKSolution;
using IKSolution_STLIter = std::vector<IKSolution>::iterator;
//!
using IKSolution_DevIter = IKSolution_STLIter;
using IKSolution_HostIter = IKSolution_STLIter;
//!





} // namespace
#endif