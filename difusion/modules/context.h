#pragma once

#include "difusion/modules/module.h"

namespace difusion {

struct Context {

  static Context &Instance() {
    static Context instance;
    return instance;
  }

  void addModule(const std::string &name, Module *m);

  Module *getModule(const std::string &name);

protected:
  std::map<std::string, Module *> modules_;

private:
  Context() = default;
  ~Context() = default;

  Context(const Context &) = delete;
  const Context &operator=(const Context &) = delete;
};

} // namespace difusion
