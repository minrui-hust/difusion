#pragma once

#include "difusion/modules/module.h"

namespace difusion {

struct PreProcessor : public Module {

  std::shared_ptr<Measurement>
  process(const std::shared_ptr<Measurement> &meas);
};

} // namespace difusion
