#pragma once

namespace difusion::preprocess {

struct Component {

  std::shared_ptr<Measurement>
  process(const std::shared_ptr<Measurement> &meas);

  virtual ~Component() = default;
};

} // namespace difusion::preprocess
