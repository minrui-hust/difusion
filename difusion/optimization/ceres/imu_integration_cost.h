#pragma once

#include "ceres/ceres.h"

#include "manif/manif.h"

#include "mfusion/dtype.h"
#include "mfusion/logging.h"
#include "mfusion/measurements/imu_measurement.h"
#include <ceres/sized_cost_function.h>

namespace mfusion {

using manif::skew;

struct ImuIntegrationCost {
  ImuIntegrationCost(const ImuIntegrationMeas::Ptr &imu_int,
                     const scalar_t gnorm)
      : imu_int(imu_int), gnorm(gnorm) {}

  template <typename T> // clang-format off
  bool operator()(const T *const tf0_data, const T *const vel0_data, const T *const bias0_data, 
                  const T *const tf1_data, const T *const vel1_data, const T *const bias1_data,
                  const T *const gravity_data, T *residual_data) const { // clang-format on
    const Eigen::Map<const manif::SE3<T>> tf0(tf0_data);
    const Eigen::Map<const Eigen::Vector3<T>> v0(vel0_data);
    const Eigen::Map<const Eigen::Vector<T, 6>> b0(bias0_data);

    const Eigen::Map<const manif::SE3<T>> tf1(tf1_data);
    const Eigen::Map<const Eigen::Vector3<T>> v1(vel1_data);
    const Eigen::Map<const Eigen::Vector<T, 6>> b1(bias1_data);

    const Eigen::Map<const Eigen::Vector3<T>> gravity(gravity_data);

    Eigen::Map<Eigen::Vector<T, 9>> residual(residual_data);

    auto r0 = tf0.r();
    auto r1 = tf1.r();
    auto p0 = tf0.t();
    auto p1 = tf1.t();

    auto g = gnorm * gravity;
    auto t = double(imu_int->dT);
    auto db = 0.5 * (b0 + b1) - imu_int->bias.cast<T>();

    auto int_mean = imu_int->int_mean.cast<T>();
    auto int_jac = imu_int->int_jac.cast<T>();

    using Bundle = decltype(int_mean);
    using Tangent = typename Bundle::Tangent;

    // measurement
    Tangent delta = int_jac * db;
    Bundle meas = int_mean.rplus(delta);

    // prediction
    Bundle pred; // clang-format off
    pred.template element<0>() = r0.icompose(r1);
    pred.template element<1>() = r0.iact(v1 - v0 + g * t);
    pred.template element<2>() = r0.iact(p1 - p0 - v0 * t + 0.5 * g * t * t);
    // clang-format on

    // residual
    Eigen::Vector<T, 9> diff = (pred - meas).coeffs();
    residual = imu_int->int_wgt.cast<T>() * diff;

    return true;
  }

  static ceres::CostFunction *Create(const ImuIntegrationMeas::Ptr &imu_int,
                                     const scalar_t G) {
    return new ceres::AutoDiffCostFunction<ImuIntegrationCost, 9, 7, 3, 6, 7, 3,
                                           6, 3>(
        new ImuIntegrationCost(imu_int, G));
  }

  ImuIntegrationMeas::Ptr imu_int;
  double gnorm;
};

struct ImuIntegrationCost2
    : public ceres::SizedCostFunction<9, 7, 3, 6, 7, 3, 6, 3> {
  ImuIntegrationCost2(const ImuIntegrationMeas::Ptr &imu_int,
                      const scalar_t gnorm)
      : imu_int(imu_int), gnorm(gnorm) {
    t = imu_int->dT;
    int_bias = imu_int->bias.cast<double>();
    int_mean = imu_int->int_mean.cast<double>();
    int_mean.element<0>().normalize();
    int_jac = imu_int->int_jac.cast<double>();
    int_wgt = imu_int->int_wgt.cast<double>();
  }

  bool Evaluate(double const *const *parameters, double *residual_data,
                double **jacobians) const override {
    const Eigen::Map<const SE3d> tf0(parameters[0]);
    const Eigen::Map<const Vector3d> v0(parameters[1]);
    const Eigen::Map<const Vector6d> b0(parameters[2]);
    const Eigen::Map<const SE3d> tf1(parameters[3]);
    const Eigen::Map<const Vector3d> v1(parameters[4]);
    const Eigen::Map<const Vector6d> b1(parameters[5]);
    const Eigen::Map<const Vector3d> gravity(parameters[6]);

    Eigen::Map<Vector9d> residual(residual_data);

    Matrix3d j_pdQ_r0, j_pdQ_r1;
    Matrix3d j_pdV_V, j_pdV_r0;
    Matrix3d j_pdP_P, j_pdP_r0;
    Matrix3d j_mdQ_ddQ;
    tl::optional<Eigen::Ref<Matrix3d>> jr_pdQ_r0, jr_pdQ_r1;
    tl::optional<Eigen::Ref<Matrix3d>> jr_pdV_V, jr_pdV_r0;
    tl::optional<Eigen::Ref<Matrix3d>> jr_pdP_P, jr_pdP_r0;
    tl::optional<Eigen::Ref<Matrix3d>> jr_mdQ_ddQ;
    if (jacobians) {
      jr_pdQ_r0 = j_pdQ_r0;
      jr_pdQ_r1 = j_pdQ_r1;
      jr_pdV_V = j_pdV_V;
      jr_pdV_r0 = j_pdV_r0;
      jr_pdP_P = j_pdP_P;
      jr_pdP_r0 = j_pdP_r0;
      jr_mdQ_ddQ = j_mdQ_ddQ;
    }

    auto r0 = tf0.r();
    auto r1 = tf1.r();
    auto p0 = tf0.t();
    auto p1 = tf1.t();
    auto g = gnorm * gravity;

    auto db = 0.5 * (b0 + b1) - int_bias;
    auto delta = decltype(int_mean)::Tangent(int_jac * db);

    // measurement
    decltype(int_mean) meas; // clang-format off
    meas.element<0>() = int_mean.element<0>().rplus(delta.element<0>(),{}, jr_mdQ_ddQ);
    meas.element<1>() = int_mean.element<1>() + delta.element<1>();
    meas.element<2>() = int_mean.element<2>() + delta.element<2>();
    // clang-format on

    // prediction
    decltype(int_mean) pred; // clang-format off
    pred.element<0>() = r0.icompose(r1, jr_pdQ_r0, jr_pdQ_r1);
    pred.element<1>() = r0.iact(v1 - v0 + g*t, jr_pdV_r0, jr_pdV_V);
    pred.element<2>() = r0.iact(p1 - p0 - v0*t + 0.5*g*t*t, jr_pdP_r0, jr_pdP_P);
    // clang-format on

    // residual
    auto diff = (pred - meas).coeffs();
    residual = int_wgt * diff;

    if (jacobians) {
      // clang-format off
      Matrixd<3, 3> j_d_pdQ =  Matrix3d::Identity() + 0.5 * skew(diff.head<3>());
      Matrixd<3, 3> j_d_mdQ = -Matrix3d::Identity() + 0.5 * skew(diff.head<3>());
      // clang-format on

      if (jacobians[0]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> jac_tf0(jacobians[0]);
        jac_tf0.rightCols<1>().setZero();
        jac_tf0.block<3,3>(0,0).setZero();
        jac_tf0.block<3,3>(0,3) = j_d_pdQ * j_pdQ_r0;
        jac_tf0.block<3,3>(3,0).setZero();
        jac_tf0.block<3,3>(3,3) = j_pdV_r0;
        jac_tf0.block<3,3>(6,0) = -j_pdP_P;
        jac_tf0.block<3,3>(6,3) = j_pdP_r0;
        jac_tf0 = int_wgt * jac_tf0;
      } // clang-format on

      if (jacobians[1]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jac_v0(jacobians[1]);
        jac_v0.block<3,3>(0,0).setZero();
        jac_v0.block<3,3>(3,0) = - j_pdV_V;
        jac_v0.block<3,3>(6,0) = - j_pdP_P * t;
        jac_v0 = int_wgt * jac_v0;
      } // clang-format on

      if (jacobians[2]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> jac_b0(jacobians[2]);
        jac_b0.block<3,3>(0,0) = j_d_mdQ * j_mdQ_ddQ * int_jac.block<3,3>(0,0) * 0.5;
        jac_b0.block<3,3>(0,3).setZero();
        jac_b0.block<3,6>(3,0) = -int_jac.block<3,6>(3,0) * 0.5;
        jac_b0.block<3,6>(6,0) = -int_jac.block<3,6>(6,0) * 0.5;
        jac_b0 = int_wgt * jac_b0;
      } // clang-format on

      if (jacobians[3]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> jac_tf1(jacobians[3]);
        jac_tf1.rightCols<1>().setZero();
        jac_tf1.block<3,3>(0,0).setZero();
        jac_tf1.block<3,3>(0,3) = j_d_pdQ * j_pdQ_r1;
        jac_tf1.block<3,3>(3,0).setZero();
        jac_tf1.block<3,3>(3,3).setZero();
        jac_tf1.block<3,3>(6,0) = j_pdP_P;
        jac_tf1.block<3,3>(6,3).setZero();
        jac_tf1 = int_wgt * jac_tf1;
      } // clang-format on

      if (jacobians[4]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jac_v1(jacobians[4]);
        jac_v1.block<3,3>(0,0).setZero();
        jac_v1.block<3,3>(3,0) = j_pdV_V;
        jac_v1.block<3,3>(6,0).setZero();
        jac_v1 = int_wgt * jac_v1;
      } // clang-format on

      if (jacobians[5]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> jac_b1(jacobians[5]);
        jac_b1.block<3,3>(0,0) = j_d_mdQ * j_mdQ_ddQ * int_jac.block<3,3>(0,0) * 0.5;
        jac_b1.block<3,3>(0,3).setZero();
        jac_b1.block<3,6>(3,0) = -int_jac.block<3,6>(3,0) * 0.5;
        jac_b1.block<3,6>(6,0) = -int_jac.block<3,6>(6,0) * 0.5;
        jac_b1 = int_wgt * jac_b1;
      } // clang-format on

      if (jacobians[6]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jac_g(jacobians[6]);
        jac_g.block<3,3>(0,0).setZero();
        jac_g.block<3,3>(3,0) = j_pdV_V * t * gnorm;
        jac_g.block<3,3>(6,0) = j_pdP_P * t * t * 0.5 * gnorm;
        jac_g = int_wgt * jac_g;
      } // clang-format on
    }

    return true;
  }

  ImuIntegrationMeas::Ptr imu_int;
  double gnorm;

  double t;
  Vector6d int_bias;
  ImuIntegrationMeas::StateBundle::LieGroupTemplate<double> int_mean;
  Matrixd<9, 6> int_jac;
  Matrix9d int_wgt;
};

struct ImuIntegrationCostBiasDiff {
  ImuIntegrationCostBiasDiff(const ImuIntegrationMeas::Ptr &imu_int)
      : imu_int(imu_int) {}

  template <typename T>
  bool operator()(const T *const bias0_data, const T *const bias1_data,
                  T *residual_data) const {
    const Eigen::Map<const Eigen::Vector<T, 6>> b0(bias0_data);
    const Eigen::Map<const Eigen::Vector<T, 6>> b1(bias1_data);

    Eigen::Map<Eigen::Vector<T, 6>> residual(residual_data);

    residual = imu_int->bias_wgt.cast<T>().asDiagonal() * (b1 - b0);

    return true;
  }

  static ceres::CostFunction *Create(const ImuIntegrationMeas::Ptr &imu_int) {
    return new ceres::AutoDiffCostFunction<ImuIntegrationCostBiasDiff, 6, 6, 6>(
        new ImuIntegrationCostBiasDiff(imu_int));
  }

  ImuIntegrationMeas::Ptr imu_int;
};

struct ImuIntegrationCostBiasDiff2 : public ceres::SizedCostFunction<6, 6, 6> {
  ImuIntegrationCostBiasDiff2(const ImuIntegrationMeas::Ptr &imu_int)
      : imu_int(imu_int) {
    bias_wgt = imu_int->bias_wgt.cast<double>();
  }

  bool Evaluate(double const *const *parameters, double *residual_data,
                double **jacobians) const {
    const Eigen::Map<const Vector6d> b0(parameters[0]);
    const Eigen::Map<const Vector6d> b1(parameters[1]);

    Eigen::Map<Vector6d> residual(residual_data);

    residual = bias_wgt.asDiagonal() * (b1 - b0);

    if (jacobians) {
      if (jacobians[0]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 6,6, Eigen::RowMajor>> jac_b0(jacobians[0]);
        jac_b0 = (-bias_wgt).asDiagonal();
      } // clang-format on

      if (jacobians[1]) { // clang-format off
        Eigen::Map<Eigen::Matrix<double, 6,6, Eigen::RowMajor>> jac_b1(jacobians[1]);
        jac_b1 = bias_wgt.asDiagonal();
      } // clang-format on
    }

    return true;
  }

  static ceres::CostFunction *Create(const ImuIntegrationMeas::Ptr &imu_int) {
    return new ceres::AutoDiffCostFunction<ImuIntegrationCostBiasDiff, 6, 6, 6>(
        new ImuIntegrationCostBiasDiff(imu_int));
  }

  ImuIntegrationMeas::Ptr imu_int;
  Vector6d bias_wgt;
};

struct ImuIntegrationCostGyroBias {
  ImuIntegrationCostGyroBias(const ImuIntegrationMeas::Ptr &imu_int,
                             const SO3 &dQ)
      : imu_int(imu_int) {
    dQ_pred = dQ.cast<double>();
    dQ_pred.normalize();
    dQ_mean = imu_int->dQ().cast<double>();
    dQ_mean.normalize();
  }

  template <typename T>
  bool operator()(const T *const bg_data, T *residual) const {
    Eigen::Map<const Eigen::Vector3<T>> bg(bg_data);
    Eigen::Map<typename manif::SO3<T>::Tangent> res(residual);

    auto dbg = bg - imu_int->bias.head<3>().cast<T>();
    auto jac = imu_int->int_jac.topLeftCorner<3, 3>().cast<T>();

    auto dQ_meas =
        dQ_mean.cast<T>() + typename manif::SO3<T>::Tangent(jac * dbg);

    res = dQ_pred.cast<T>() - dQ_meas;

    return true;
  }

  static ceres::CostFunction *Create(const ImuIntegrationMeas::Ptr &imu_int,
                                     const SO3 &dQ) {
    return new ceres::AutoDiffCostFunction<ImuIntegrationCostGyroBias, 3, 3>(
        new ImuIntegrationCostGyroBias(imu_int, dQ));
  }

  ImuIntegrationMeas::Ptr imu_int;
  SO3d dQ_pred;
  SO3d dQ_mean;
};

struct ImuIntegrationCostScale {
  ImuIntegrationCostScale(const ImuIntegrationMeas::Ptr &imu_int,
                          const SO3 &q_i_w, const Vector3 &p_i_j,
                          const Vector3 &e_i_j, const scalar_t gnorm)
      : imu_int(imu_int), q_i_w(q_i_w), p_i_j(p_i_j), e_i_j(e_i_j),
        gnorm(gnorm) {}

  template <typename T>
  bool operator()(const T *const vi_data, const T *const vj_data,
                  const T *const gravity_data, const T *const scale_data,
                  T *residual_data) const {
    const Eigen::Map<const Eigen::Vector3<T>> vi(vi_data);
    const Eigen::Map<const Eigen::Vector3<T>> vj(vj_data);
    const Eigen::Map<const Eigen::Vector3<T>> gravity(gravity_data);
    const auto &s = *scale_data;

    Eigen::Map<Eigen::Vector<T, 6>> residual(residual_data);

    auto t = double(imu_int->dT);
    auto g = gnorm * gravity;
    auto dV = imu_int->dV().cast<T>();
    auto dP = imu_int->dP().cast<T>();
    auto q = q_i_w.cast<T>();
    auto pij = p_i_j.cast<T>();
    auto eij = e_i_j.cast<T>();

    residual.template head<3>() = dV - q.act(vj + g * t - vi);

    residual.template tail<3>() =
        dP - q.act(eij + s * pij + 0.5 * g * t * t - vi * t);

    return true;
  }

  static ceres::CostFunction *Create(const ImuIntegrationMeas::Ptr &imu_int,
                                     const SO3 &q_i_w, const Vector3 &p_i_j,
                                     const Vector3 &e_i_j,
                                     const scalar_t gnorm) {
    return new ceres::AutoDiffCostFunction<ImuIntegrationCostScale, 6, 3, 3, 3,
                                           1>(
        new ImuIntegrationCostScale(imu_int, q_i_w, p_i_j, e_i_j, gnorm));
  }

  ImuIntegrationMeas::Ptr imu_int;
  SO3 q_i_w;
  Vector3 p_i_j;
  Vector3 e_i_j;
  double gnorm;
};

} // namespace mfusion
