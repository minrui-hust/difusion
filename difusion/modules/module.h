#pragma once

#include "yaml-cpp/yaml.h"

#include "difusion/dtype.h"

namespace difusion {

struct Module {
  virtual void configure(const YAML::Node &cfg) = 0;

  // base class should set destructor virtual
  virtual ~Module() = default;
};

} // namespace difusion
