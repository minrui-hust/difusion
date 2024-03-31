#pragma once

#include "difusion/measurements/measurement.h"
#include "difusion/modules/module.h"
#include "difusion/states/state.h"

namespace difusion {

struct Estimator : public Module {
  Estimator(const YAML::Node &cfg);

  // handle all kinds of measurment input
  virtual bool handleMeasurement(const std::shared_ptr<Measurement> &meas) = 0;

  // callback will be called when state updated
  void registerCallback(
      const std::function<void(const std::shared_ptr<State> &)> &cb) {
    on_state_update_ = cb;
  }

  static std::unique_ptr<Estimator> create(const YAML::Node &cfg);

protected:
  std::function<void(const std::shared_ptr<State> &)> on_state_update_;
};

} // namespace difusion
