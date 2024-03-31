#pragma once

#include "difusion/measurements/measurement.h"
#include "difusion/modules/module.h"

namespace difusion {

struct Localmap : public Module {
  Localmap(const YAML::Node &cfg);

  virtual bool push(const std::shared_ptr<Measurement> &meas) = 0;

  virtual bool getClosestPlanePoints(const Vector3 pos,
                                     std::vector<Vector3> &nbs) = 0;

  virtual bool getClosestEdgePoints(const Vector3 pos,
                                    std::vector<Vector3> &nbs) = 0;

  virtual bool getClosestRawPoints(const Vector3 pos,
                                   std::vector<Vector3> &nbs) = 0;

  static std::unique_ptr<Localmap> create(const YAML::Node &cfg);
};

} // namespace difusion
