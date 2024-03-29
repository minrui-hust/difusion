#pragma once

#include "measurement.h"

namespace difusion {

struct ImuMeasurement : public Measurement {};

struct ImuIntegrationMeas : public Measurement {};

} // namespace difusion
