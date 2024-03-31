#pragma once

#include "difusion/modules/module.h"
#include "difusion/states/state.h"

namespace difusion {

struct StateInterpolator : public Module {
  StateInterpolator(const YAML::Node &cfg);

  // push state into state buffer
  bool insert(const std::shared_ptr<State> &state);

  // get the state at specific time
  std::shared_ptr<State> get(const Time);

  static std::unique_ptr<StateInterpolator> create(const YAML::Node &cfg);
};

} // namespace difusion
