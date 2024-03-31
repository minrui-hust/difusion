#pragma once

#include "measurement.h"

namespace difusion {

struct ImuMeasurement : public Measurement {
  Vector3 w;
  Vector3 a;
};

} // namespace difusion
