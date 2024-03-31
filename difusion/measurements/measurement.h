#pragma once

#include "difusion/dtype.h"

namespace difusion {

struct MeasurementHeader {
  int id;     // sensor id
  Time stamp; // sample time in us
};

struct Measurement {
  MeasurementHeader header;

  // base class should set destructor virtual
  virtual ~Measurement() = default;
};

} // namespace difusion
