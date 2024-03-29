#pragma once

#include <functional>
#include <map>
#include <vector>

#include <cstdint>
#include <manif/manif.h>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "Eigen/src/Geometry/Quaternion.h"
#include "manif/SE3.h"

namespace difusion {

using scalar_t = double;

using Time = int64_t;

template <size_t _dim> using Vector = Eigen::Vector<scalar_t, _dim>;

template <size_t _dim> using Vectord = Eigen::Vector<double, _dim>;

using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Vector5 = Vector<5>;
using Vector6 = Vector<6>;

using Vector2d = Vectord<2>;
using Vector3d = Vectord<3>;
using Vector4d = Vectord<4>;
using Vector5d = Vectord<5>;
using Vector6d = Vectord<6>;
using Vector9d = Vectord<9>;

template <size_t _rows, size_t _cols>
using Matrix = Eigen::Matrix<scalar_t, _rows, _cols>;

template <size_t _rows, size_t _cols>
using Matrixd = Eigen::Matrix<double, _rows, _cols>;

using Matrix2 = Matrix<2, 2>;
using Matrix3 = Matrix<3, 3>;
using Matrix4 = Matrix<4, 4>;
using Matrix5 = Matrix<5, 5>;
using Matrix6 = Matrix<6, 6>;
using Matrix9 = Matrix<9, 9>;

using Matrix2d = Matrixd<2, 2>;
using Matrix3d = Matrixd<3, 3>;
using Matrix4d = Matrixd<4, 4>;
using Matrix5d = Matrixd<5, 5>;
using Matrix6d = Matrixd<6, 6>;
using Matrix9d = Matrixd<9, 9>;

using Quaternion = Eigen::Quaternion<scalar_t>;

using SO2 = manif::SO2<scalar_t>;
using SE2 = manif::SE2<scalar_t>;
using SO3 = manif::SO3<scalar_t>;
using SE3 = manif::SE3<scalar_t>;

using SO2d = manif::SO2<double>;
using SE2d = manif::SE2<double>;
using SO3d = manif::SO3<double>;
using SE3d = manif::SE3<double>;

using R2 = manif::Rn<scalar_t, 2>;
using R3 = manif::Rn<scalar_t, 3>;
using R4 = manif::Rn<scalar_t, 4>;
using R5 = manif::Rn<scalar_t, 5>;
using R6 = manif::Rn<scalar_t, 6>;

using R2d = manif::Rn<double, 2>;
using R3d = manif::Rn<double, 3>;
using R4d = manif::Rn<double, 4>;
using R5d = manif::Rn<double, 5>;
using R6d = manif::Rn<double, 6>;

template <typename _First, typename... _Remains>
using Bundle =
    manif::Bundle<typename _First::Scalar, _First::template LieGroupTemplate,
                  _Remains::template LieGroupTemplate...>;

#define NANOS_PER_SECOND 1000000000ul
#define MICROS_PER_SECOND 1000000ul
#define MILLIS_PER_SECOND 1000ul
#define NANOS_PER_MICRO 1000ul

} // namespace difusion
