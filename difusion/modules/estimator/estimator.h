#pragma once

#include "difusion/modules/module.h"

namespace difusion {

struct Estimator : public Module {

  bool handleMeasurement(const std::shared_ptr<Measurement> &meas);
};

} // namespace difusion
