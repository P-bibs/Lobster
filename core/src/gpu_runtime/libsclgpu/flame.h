#pragma once
#include <exception>
#include <string>

#include "utils.h"

extern "C" {
void flame_start(const char *name);
void flame_end(const char *name);
void flame_dump(const char *name);
void flame_dump_html(const char *name);
void flame_debug();
void flame_dump_stdout();
void flame_clear();
}

class TraceGuard {
 private:
  std::string name_;
  bool stopped_;

 public:
  TraceGuard(const char *name) : name_(name), stopped_(false) {
    flame_start(name_.c_str());
  }
  void stop() {
    if (!stopped_) {
      stopped_ = true;
      flame_end(name_.c_str());
    }
  }
  ~TraceGuard() { this->stop(); }
};

#ifdef TRACE
#define TRACE_START(name) TraceGuard name(#name)
#define TRACE_END(name) name.stop()
#else
#define TRACE_START(name)
#define TRACE_END(name)
#endif
