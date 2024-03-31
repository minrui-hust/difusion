#pragma once

#include "yaml-cpp/yaml.h"

#include "difusion/dtype.h"

namespace difusion {

struct Module {
  Module(const YAML::Node &cfg) : config_(cfg) {}

  // base class should set destructor virtual
  virtual ~Module() = default;

protected:
  YAML::Node config_;
};

} // namespace difusion
