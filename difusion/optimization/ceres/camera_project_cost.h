#pragma once

#include "ceres/ceres.h"

#include "manif/manif.h"

#include "mfusion/dtype.h"
#include "mfusion/logging.h"
#include "mfusion/utils.h"

namespace mfusion {

struct RelativeCameraProjectCost {
  RelativeCameraProjectCost(const Vector2 &uv0, const Vector2 &uv1,
                            const SE3 &tf_base_cam, const Vector2 &wgt) {
    this->xyz.head<2>() = uv0.cast<double>();
    this->xyz.z() = 1;
    this->uv = uv1.cast<double>();
    this->wgt = wgt.cast<double>();
    this->tf_b_c = tf_base_cam.cast<double>();
  }

  template <typename T>
  bool operator()(const T *const tf0_data, const T *const tf1_data,
                  const T *const depth_inv_data, T *residual_data) const {
    // Timer timer("RelativeCameraProjectCost.JET");
    const Eigen::Map<const manif::SE3<T>> tf0(tf0_data);
    const Eigen::Map<const manif::SE3<T>> tf1(tf1_data);
    const T &inv_depth = *depth_inv_data;

    Eigen::Map<Eigen::Vector2<T>> residual(residual_data);

    manif::SE3<T> extrinsic = tf_b_c.cast<T>();

    Eigen::Vector3<T> p_cam0 = xyz.cast<T>() / inv_depth;

    Eigen::Vector3<T> p_imu0 = extrinsic.act(p_cam0);
    Eigen::Vector3<T> p_map = tf0.act(p_imu0);
    Eigen::Vector3<T> p_imu1 = tf1.iact(p_map);
    Eigen::Vector3<T> p_cam1 = extrinsic.iact(p_imu1);

    residual[0] = wgt[0] * (uv[0] - p_cam1[0] / p_cam1[2]);
    residual[1] = wgt[1] * (uv[1] - p_cam1[1] / p_cam1[2]);

    return true;
  }

  static ceres::CostFunction *Create(const Vector2 &uv0, const Vector2 &uv1,
                                     const SE3 &tf_base_cam,
                                     const Vector2 &wgt) {
    return new ceres::AutoDiffCostFunction<RelativeCameraProjectCost, 2, 7, 7,
                                           1>(
        new RelativeCameraProjectCost(uv0, uv1, tf_base_cam, wgt));
  }

  Vector3d xyz;
  Vector2d wgt;
  Vector2d uv;
  SE3d tf_b_c;
};

struct RelativeCameraProjectCost2
    : public ceres::SizedCostFunction<2, 7, 7, 1> {
  RelativeCameraProjectCost2(const Vector2 &uv0, const Vector2 &uv1,
                             const SE3 &tf_base_cam, const Vector2 &wgt) {
    this->xyz.head<2>() = uv0.cast<double>();
    this->xyz.z() = 1;
    this->uv = uv1.cast<double>();
    this->wgt = wgt.cast<double>();
    this->tf_b_c = tf_base_cam.cast<double>();
    Rbc = tf_b_c.rotation();
    Rcb = Rbc.transpose();
  }

  virtual bool Evaluate(double const *const *parameters, double *residual_data,
                        double **jacobians) const override {
    const Eigen::Map<const SE3d> tf0(parameters[0]);
    const Eigen::Map<const SE3d> tf1(parameters[1]);
    const double &depth = 1 / (*parameters[2]);

    Eigen::Map<Vector2d> residual(residual_data);

    Vector3d p_cam0 = xyz * depth;
    Vector3d p_imu0 = tf_b_c.act(p_cam0);
    Vector3d p_map = tf0.act(p_imu0);
    Vector3d p_imu1 = tf1.iact(p_map);
    Vector3d p_cam1 = tf_b_c.iact(p_imu1);

    double inv_depth1 = 1 / p_cam1[2];
    Vector2d proj = p_cam1.head<2>() * inv_depth1;

    residual = wgt.asDiagonal() * (uv - proj);

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> j_r_pcam1;
      j_r_pcam1.leftCols<2>() = -Matrix2d::Identity() * inv_depth1;
      j_r_pcam1.rightCols<1>() = proj * inv_depth1;

      auto Rw0 = tf0.rotation();
      auto Rw1 = tf1.rotation();
      auto R1w = Rw1.transpose();
      auto t1 = tf1.t();

      Matrixd<2, 3> j_r_pimu1 = wgt.asDiagonal() * j_r_pcam1 * Rbc.transpose();
      Matrixd<2, 3> j_r_pmap = j_r_pimu1 * R1w;
      Matrixd<2, 3> j_r_pimu0 = j_r_pmap * Rw0;

      if (jacobians[0]) { // J_residual_tf0
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jac_tf0(
            jacobians[0]);
        jac_tf0.leftCols<3>() = j_r_pmap;
        jac_tf0.middleCols<3>(3) = -j_r_pimu0 * manif::skew(p_imu0);
        jac_tf0.rightCols<1>().setZero();
      }
      if (jacobians[1]) { // J_residual_tf1
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jac_tf1(
            jacobians[1]);
        jac_tf1.leftCols<3>() = -j_r_pmap;
        jac_tf1.middleCols<3>(3) = j_r_pmap * manif::skew(p_map - t1) * Rw1;
        jac_tf1.rightCols<1>().setZero();
      }
      if (jacobians[2]) { // J_residual_idepth
        Eigen::Map<Eigen::Matrix<double, 2, 1>> jac_idep(jacobians[2]);
        jac_idep = -j_r_pimu0 * Rbc * p_cam0 * depth;
      }
    }

    return true;
  }

  static ceres::CostFunction *Create(const Vector2 &uv0, const Vector2 &uv1,
                                     const SE3 &tf_base_cam,
                                     const Vector2 &wgt) {
    return new RelativeCameraProjectCost2(uv0, uv1, tf_base_cam, wgt);
  }

  Vector3d xyz;
  Vector2d wgt;
  Vector2d uv;
  SE3d tf_b_c;
  Matrix3d Rbc;
  Matrix3d Rcb;
};

struct CameraProjectCost {
  CameraProjectCost(const Vector2 &uv) : uv(uv.cast<double>()) {}

  template <typename T>
  bool operator()(const T *const camera_R, const T *const camera_t,
                  const T *const p_world, T *residuals) const {
    // Timer timer("CameraProjectCost.JET");
    const Eigen::Map<const manif::SO3<T>> R(camera_R);
    const Eigen::Map<const Eigen::Vector3<T>> t(camera_t);
    const Eigen::Map<const Eigen::Vector3<T>> p(p_world);

    Eigen::Map<Eigen::Vector2<T>> r(residuals);

    Eigen::Vector3<T> proj = R.act(p) + t;

    r[0] = uv[0] - proj[0] / proj[2];
    r[1] = uv[1] - proj[1] / proj[2];

    return true;
  }

  static ceres::CostFunction *Create(const Vector2 &uv) {
    return new ceres::AutoDiffCostFunction<CameraProjectCost, 2, 4, 3, 3>(
        new CameraProjectCost(uv));
  }

  Vector2d uv;
};

struct CameraProjectCost2 : public ceres::SizedCostFunction<2, 4, 3, 3> {
  CameraProjectCost2(const Vector2 &_uv) : uv(_uv.cast<double>()) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const override {
    const Eigen::Map<const manif::SO3d> R(parameters[0]);
    const Eigen::Map<const Eigen::Vector3d> t(parameters[1]);
    const Eigen::Map<const Eigen::Vector3d> w(parameters[2]);

    Eigen::Map<Eigen::Vector2d> r(residuals);

    Vector3d p = R.act(w) + t;

    double idepth = 1 / p[2];
    Vector2d proj = p.head<2>() * idepth;

    r = uv - proj;

    if (jacobians) {
      Eigen::Matrix<double, 2, 3> j_r_p;
      // clang-format off
      j_r_p << -idepth,     0   , proj[0] * idepth, 
                  0,     -idepth, proj[1] * idepth;
      // clang-format on

      Matrixd<2, 3> j_r_w = j_r_p * R.rotation();

      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jac0(
            jacobians[0]);
        jac0.leftCols<3>() = -j_r_w * manif::skew(w);
        jac0.rightCols<1>().setZero();
      }

      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac1(
            jacobians[1]);
        jac1 = j_r_p;
      }

      if (jacobians[2]) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac2(
            jacobians[2]);
        jac2 = j_r_w;
      }
    }

    return true;
  }

  Vector2d uv;
};

} // namespace mfusion
