#include "difusion/modules/context.h"

#include "fuser.h"

namespace difusion {

Fuser::Fuser(const YAML::Node &cfg) : Module(cfg) {
  // create submodules
  preprocessor_ = Preprocessor::create(cfg["preprocessor"]);
  estimator_ = Estimator::create(cfg["estimator"]);
  localmap_ = Localmap::create(cfg["localmap"]);
  state_interp_ = StateInterpolator::create(cfg["state_interp"]);

  // register modules in context (other module or component may use)
  Context::Instance().addModule("localmap", localmap_.get());
  Context::Instance().addModule("state_interp", state_interp_.get());

  // estimator callback
  estimator_->registerCallback(
      std::bind(&Fuser::handleEstimatorUpdate, this, std::placeholders::_1));
}

bool Fuser::handleMeasurement(const std::shared_ptr<Measurement> &meas) {
  // TODO
  return true;
}

void Fuser::handleEstimatorUpdate(const std::shared_ptr<State> &s) {
  // TODO
}

} // namespace difusion
