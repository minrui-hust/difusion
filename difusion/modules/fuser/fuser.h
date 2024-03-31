#pragma once

#include "difusion/measurements/measurement.h"
#include "difusion/modules/module.h"
#include "difusion/states/state.h"

#include "difusion/modules/estimator/estimator.h"
#include "difusion/modules/localmap/localmap.h"
#include "difusion/modules/preprocess/preprocessor.h"
#include "difusion/modules/state_interp/state_interp.h"

namespace difusion {

struct Fuser : public Module {
  Fuser(const YAML::Node &cfg);

  bool handleMeasurement(const std::shared_ptr<Measurement> &meas);

  // callback will be called when state updated
  void registerCallback(
      const std::function<void(const std::shared_ptr<State> &)> &cb) {
    on_state_update_ = cb;
  }

protected:
  void handleEstimatorUpdate(const std::shared_ptr<State> &s);

protected:
  std::unique_ptr<Preprocessor> preprocessor_;
  std::unique_ptr<Estimator> estimator_;
  std::unique_ptr<Localmap> localmap_;
  std::unique_ptr<StateInterpolator> state_interp_;

  std::function<void(const std::shared_ptr<State> &)> on_state_update_;
};

} // namespace difusion
