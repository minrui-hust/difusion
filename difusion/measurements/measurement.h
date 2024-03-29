#pragma once

#include <memory>

#include "difusion/dtype.h"

namespace difusion {

struct MeasurementHeader {
  int id;     // sensor id
  Time stamp; // sample time in us
};

struct Measurement {
  MeasurementHeader header;

  // cast to derived class object
  template <typename Derived> Derived &as() {
    return static_cast<Derived &>(*this);
  }

  // cast to derived class object const version
  template <typename Derived> const Derived &as() const {
    return static_cast<const Derived &>(*this);
  }

  // base class should set destructor virtual
  virtual ~Measurement() = default;
};

} // namespace difusion
