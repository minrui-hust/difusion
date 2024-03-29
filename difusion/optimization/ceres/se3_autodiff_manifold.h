#pragma once

#include "ceres/ceres.h"

#include "manif/manif.h"

#include "mfusion/dtype.h"
#include "mfusion/logging.h"

namespace mfusion {

class SE3ManifoldFunctor {

public:
  SE3ManifoldFunctor() = default;
  virtual ~SE3ManifoldFunctor() = default;

  template <typename T>
  bool Plus(const T *x_data, const T *delta_data, T *x_plus_delta_data) const {
    const Eigen::Map<const manif::SE3<T>> x(x_data);
    const Eigen::Map<const manif::SE3Tangent<T>> delta(delta_data);

    Eigen::Map<manif::SE3<T>> x_plus_delta(x_plus_delta_data);

    x_plus_delta.t() = x.t() + delta.lin();
    x_plus_delta.r() = x.r() + delta.asSO3();

    return true;
  }

  template <typename T>
  bool Minus(const T *y_data, const T *x_data, T *y_minus_x_data) const {
    const Eigen::Map<const manif::SE3<T>> y(y_data);
    const Eigen::Map<const manif::SE3<T>> x(x_data);

    Eigen::Map<manif::SE3Tangent<T>> y_minus_x(y_minus_x_data);

    y_minus_x.lin() = y.t() - x.t();
    y_minus_x.asSO3() = y.r() - x.r();

    return true;
  }
};

} // namespace mfusion
