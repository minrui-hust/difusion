#pragma once

#include "ceres/ceres.h"

#include "manif/manif.h"

#include "mfusion/dtype.h"
#include "mfusion/logging.h"

namespace mfusion {

struct SE3DummyParameterization : public ceres::LocalParameterization {

  virtual bool Plus(const double *x_data, const double *delta_data,
                    double *x_plus_delta_data) const {
    const Eigen::Map<const SE3d> x(x_data);
    const Eigen::Map<const SE3d::Tangent> delta(delta_data);
    Eigen::Map<SE3d> x_plus_delta(x_plus_delta_data);
    // use bundle plus rather than SE3 plus
    x_plus_delta.t() = x.t() + delta.lin();
    x_plus_delta.r() = x.r() + delta.asSO3();
    return true;
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  virtual bool ComputeJacobian(const double *x, double *jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jac(jacobian);
    jac.topRows<6>().setIdentity();
    jac.bottomRows<1>().setZero();
    return true;
  }

  // Size of x.
  virtual int GlobalSize() const { return 7; }

  // Size of delta.
  virtual int LocalSize() const { return 6; }
};
} // namespace mfusion
