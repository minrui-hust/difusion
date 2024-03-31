#pragma once

#include "difusion/measurements/measurement.h"
#include "difusion/modules/module.h"

namespace difusion::preprocess {

struct Component : public Module {
  Component(const YAML::Node &cfg);

  virtual std::shared_ptr<Measurement>
  process(const std::shared_ptr<Measurement> &meas) = 0;

  virtual ~Component() = default;
};

} // namespace difusion::preprocess
