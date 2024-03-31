#pragma once

#include "difusion/dtype.h"

namespace difusion {

struct State {
  Time stamp; // time stamp in us

  virtual Vector3 position() = 0;
  virtual Vector3 linear_vel() = 0;
  virtual Vector3 linear_acc() = 0;

  virtual SO3 attitude() = 0;
  virtual Vector3 angular_vel() = 0;

  virtual SE3 pose() = 0;
  virtual Vector6 vel() = 0;

  virtual ~State() = default;
};

} // namespace difusion
