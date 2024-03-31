#pragma once

#include "difusion/measurements/measurement.h"
#include "difusion/modules/module.h"

namespace difusion {

struct Preprocessor : public Module {
  Preprocessor(const YAML::Node &cfg);

  std::shared_ptr<Measurement>
  process(const std::shared_ptr<Measurement> &meas);

  static std::unique_ptr<Preprocessor> create(const YAML::Node &cfg);
};

} // namespace difusion
