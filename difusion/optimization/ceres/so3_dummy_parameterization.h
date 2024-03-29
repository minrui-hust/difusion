#pragma once

#include "ceres/ceres.h"

#include "manif/manif.h"

#include "mfusion/dtype.h"
#include "mfusion/logging.h"

namespace mfusion {

struct SO3DummyParameterization : public ceres::LocalParameterization {

  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const {
    Eigen::Map<const manif::SO3d> R(x);
    Eigen::Map<const manif::SO3d::Tangent> d(delta);
    Eigen::Map<manif::SO3d> m(x_plus_delta);
    m = R.rplus(d);
    return true;
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  virtual bool ComputeJacobian(const double *x, double *jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jac(jacobian);
    jac.topRows<3>().setIdentity();
    jac.bottomRows<1>().setZero();
    return true;
  }

  // Size of x.
  virtual int GlobalSize() const { return 4; }

  // Size of delta.
  virtual int LocalSize() const { return 3; }
};
} // namespace mfusion
