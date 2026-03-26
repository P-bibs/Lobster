#pragma once
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <variant>

#include "flame.h"
#include "utils.h"

struct ForeignAlloc {
  template <typename T, typename... Args>
  static T *make(Args &&...) {
    throw std::runtime_error("Cannot allocate foreign memory");
  }

  template <typename T>
  static void destroy(T *ptr) {
    (void)ptr;
    // leak memory since we don't know what allocator was used
  }

  template <typename T>
  static T *malloc(size_t size) {
    (void)size;
    throw std::runtime_error("Cannot allocate foreign memory");
  }
};

struct HostAlloc {
  static std::unordered_map<void *, std::tuple<size_t, std::string>> &
  allocations() {
    static std::unordered_map<void *, std::tuple<size_t, std::string>>
        allocations;
    return allocations;
  }
  static std::multimap<void *, size_t> &all_allocations() {
    static std::multimap<void *, size_t> all_allocations;
    return all_allocations;
  }

  template <typename T, typename... Args>
  static T *make(Args &&...args) {
    auto ptr = static_cast<T *>(std::malloc(sizeof(T)));
    new (ptr) T(std::forward<Args>(args)...);
    if constexpr (TRACK_ALLOC) {
      allocations()[ptr] = std::make_tuple(sizeof(T), typeid(T).name());
      all_allocations().insert({ptr, sizeof(T)});
    }
    return ptr;
  }

  template <typename T>
  static void destroy(T *ptr) {
    std::free(ptr);
    if constexpr (TRACK_ALLOC) {
      // if (all_allocations().find(ptr) != all_allocations().end() &&
      //     allocations().find(ptr) == allocations().end()) {
      //   throw std::runtime_error("HostAlloc: double free");
      // }
      // if (all_allocations().find(ptr) == all_allocations().end()) {
      //   throw std::runtime_error("HostAlloc: freeing foreign memory");
      // }
      allocations().erase(ptr);
    }
  }

  template <typename T>
  static T *malloc(size_t size) {
    T *ptr = static_cast<T *>(std::malloc(sizeof(T) * size));
    if constexpr (TRACK_ALLOC) {
      allocations()[ptr] = std::make_tuple(sizeof(T) * size, typeid(T).name());
      all_allocations().insert({ptr, sizeof(T) * size});
    }
    return ptr;
  }
};

bool use_arena();
size_t &iter_allocs();

struct ArenaAlloc {
 private:
  char *start_;
  char *current_;
  size_t size_;

 public:
  explicit ArenaAlloc(size_t size) {
    std::cout << "Creating new Arena allocator" << std::endl;
    cudaCheck(cudaMalloc(&start_, size));
    current_ = start_;
    size_ = size;
    if (SAFETY) {
      cudaCheck(cudaMemset(this->start_, -1, this->size_));
    }
  }
  ~ArenaAlloc() { cudaCheck(cudaFree(start_)); }
  ArenaAlloc(ArenaAlloc &) = delete;
  ArenaAlloc &operator=(ArenaAlloc &) = delete;
  ArenaAlloc(ArenaAlloc &&other) {
    start_ = other.start_;
    current_ = other.current_;
    size_ = other.size_;
    other.start_ = nullptr;
    other.current_ = nullptr;
    other.size_ = 0;
  }
  ArenaAlloc &operator=(ArenaAlloc &&other) {
    if (this == &other) {
      return *this;
    }
    start_ = other.start_;
    current_ = other.current_;
    size_ = other.size_;
    other.start_ = nullptr;
    other.current_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  void reset() {
    current_ = start_;
    if (SAFETY) {
      cudaCheck(cudaMemset(this->start_, -1, this->size_));
    }
  }

  size_t used() { return current_ - start_; }

  template <typename T, typename... Args>
  T *make(Args &&...args) {
    TRACE_START(ArenaAlloc__make);
    T *ptr = this->malloc<T>(1);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }

  template <typename T>
  void destroy(T *) {
    // no-op
    // std::cout << "ArenaAlloc::free " << (size_t)ptr << std::endl;
  }

  template <typename T>
  T *malloc(size_t size) {
    TRACE_START(ArenaAlloc__malloc);
    T *ptr;
    ptr = (T *)this->allocate(sizeof(T) * size);
    // std::cout << "ArenaAlloc::malloc " << (size_t)ptr << " " << sizeof(T) *
    // size
    //           << std::endl;
    return ptr;
  }

 private:
  void *allocate(size_t size);
};

struct LeapfrogAlloc {
 private:
   char *start_;
   size_t size_;

   char *follower_start_;
   char *follower_end_;
   char *leader_start_;
   char *leader_end_;

 public:
  explicit LeapfrogAlloc(size_t size) {
    std::cout << "Creating new Leapfrog allocator" << std::endl;
    cudaCheck(cudaMalloc(&start_, size));

    size_ = size;
    follower_start_ = start_;
    follower_end_ = start_+8;
    leader_start_ = follower_end_;
    leader_end_ = follower_end_;

    if (SAFETY) {
      cudaCheck(cudaMemset(this->start_, -1, this->size_));
    }
  }
  ~LeapfrogAlloc() { cudaCheck(cudaFree(start_)); }
  LeapfrogAlloc(LeapfrogAlloc &) = delete;
  LeapfrogAlloc &operator=(LeapfrogAlloc &) = delete;
  LeapfrogAlloc(LeapfrogAlloc &&other) {
    start_ = other.start_;
    size_ = other.size_;
    follower_start_ = other.follower_start_;
    follower_end_ = other.follower_end_;
    leader_start_ = other.leader_start_;
    leader_end_ = other.leader_end_;

    other.start_ = nullptr;
    other.size_ = 0;
    other.follower_start_ = nullptr;
    other.follower_end_ = nullptr;
    other.leader_start_ = nullptr;
    other.leader_end_ = nullptr;
  }
  LeapfrogAlloc &operator=(LeapfrogAlloc &&other) {
    if (this == &other) {
      return *this;
    }
    start_ = other.start_;
    size_ = other.size_;
    follower_start_ = other.follower_start_;
    follower_end_ = other.follower_end_;
    leader_start_ = other.leader_start_;
    leader_end_ = other.leader_end_;

    other.start_ = nullptr;
    other.size_ = 0;
    other.follower_start_ = nullptr;
    other.follower_end_ = nullptr;
    other.leader_start_ = nullptr;
    other.leader_end_ = nullptr;
    return *this;
  }
  void print();

  void new_leader();
  void forget_follower();

  size_t follower_size();
  size_t leader_size();

  char *end();
  void validate();

  template <typename T, typename... Args>
  T *make(Args &&...args) {
    TRACE_START(ArenaAlloc__make);
    T *ptr = this->malloc<T>(1);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
  template <typename T>
  void destroy(T *) {
  }
  template <typename T>
  T *malloc(size_t size) {
    TRACE_START(ArenaAlloc__malloc);
    T *ptr;
    ptr = (T *)this->allocate(sizeof(T) * size);
    return ptr;
  }

 private:
  void *allocate(size_t size);
};

class DefaultAlloc {
 public:
  DefaultAlloc() {}

  template <typename T, typename... Args>
  T *make(Args &&...args) {
    TRACE_START(DefaultAlloc__make);
    T *ptr = this->malloc<T>(1);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }

  template <typename T>
  void destroy(T *ptr) {
    TRACE_START(DefaultAlloc__destroy);
    // std::cout << "DefaultAlloc::free " << (size_t)ptr << std::endl;
    cudaCheck(cudaFreeAsync(ptr, 0));
  }

  template <typename T>
  T *malloc(size_t size) {
    TRACE_START(DefaultAlloc__malloc);
    T *ptr;
    cudaCheck(cudaMallocAsync(&ptr, sizeof(T) * size, 0));
    iter_allocs() += sizeof(T) * size;
    // std::cout << "DefaultAlloc::malloc " << (size_t)ptr << " " << sizeof(T) *
    // size
    //           << std::endl;
    return ptr;
  }
};

class Allocator;

Allocator &lobster_global_allocator();

class Allocator {
 private:
  std::variant<DefaultAlloc, ArenaAlloc *, LeapfrogAlloc *> alloc_;

 public:
  Allocator() : alloc_(lobster_global_allocator().alloc_) {
    if (std::holds_alternative<ArenaAlloc *>(alloc_)) {
      //std::cout << "Constructed Allocator with ArenaAlloc" << std::endl;
    } else if (std::holds_alternative<DefaultAlloc>(alloc_)) {
      //std::cout << "Constructed Allocator with DefaultAlloc" << std::endl;
    } else if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      //std::cout << "Constructed Allocator with LeapfrogAlloc" << std::endl;
    } else {
      PANIC("Unknown allocator type");
    }
  }
  explicit Allocator(DefaultAlloc) : alloc_(DefaultAlloc()) {}
  explicit Allocator(ArenaAlloc *alloc) : alloc_(alloc) {}
  explicit Allocator(LeapfrogAlloc *alloc) : alloc_(alloc) {}

  template <typename T, typename... Args>
  T *make(Args &&...args) {
    if (std::holds_alternative<DefaultAlloc>(alloc_)) {
      return std::get<DefaultAlloc>(alloc_).make<T>(
          std::forward<Args>(args)...);
    } else if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      return std::get<LeapfrogAlloc *>(alloc_)->make<T>(
          std::forward<Args>(args)...);
    } else {
      return std::get<ArenaAlloc *>(alloc_)->make<T>(
          std::forward<Args>(args)...);
    }
  }
  template <typename T>
  void destroy(T *ptr) {
    if (std::holds_alternative<DefaultAlloc>(alloc_)) {
      std::get<DefaultAlloc>(alloc_).destroy(ptr);
    } else if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      std::get<LeapfrogAlloc *>(alloc_)->destroy(ptr);
    } else {
      std::get<ArenaAlloc *>(alloc_)->destroy(ptr);
    }
  }
  template <typename T>
  T *malloc(size_t size) {
    if (std::holds_alternative<DefaultAlloc>(alloc_)) {
      return std::get<DefaultAlloc>(alloc_).malloc<T>(size);
    } else if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      return std::get<LeapfrogAlloc *>(alloc_)->malloc<T>(size);
    } else {
      return std::get<ArenaAlloc *>(alloc_)->malloc<T>(size);
    }
  }

  void new_leader() {
    if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      std::get<LeapfrogAlloc *>(alloc_)->new_leader();
    }
  }
  void forget_follower() {
    if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      std::get<LeapfrogAlloc *>(alloc_)->forget_follower();
    }
  }
  LeapfrogAlloc *get_leapfrog() {
    if (std::holds_alternative<LeapfrogAlloc *>(alloc_)) {
      return std::get<LeapfrogAlloc *>(alloc_);
    } else {
      return nullptr;
    }
  }

  void reset() {
    if (std::holds_alternative<ArenaAlloc *>(alloc_)) {
      std::get<ArenaAlloc *>(alloc_)->reset();
    }
  }
  size_t used() {
    if (std::holds_alternative<ArenaAlloc *>(alloc_)) {
      return std::get<ArenaAlloc *>(alloc_)->used();
    } else {
      return 0;
    }
  }
};

class DeviceAlloc {
 public:
  static Allocator singleton_alloc();

  template <typename T, typename... Args>
  T *make(Args &&...args) {
    auto result = singleton_alloc().make<T>(std::forward<Args>(args)...);
    if (TRACK_ALLOC) {
      auto i = id();
      allocations()[result] = std::make_tuple(i, sizeof(T), typeid(T).name());
      current_allocations()[result] =
          std::make_tuple(i, sizeof(T), typeid(T).name());
    }
    // std::cout << "make: " << (size_t)result << " " << i << " " << sizeof(T)
    // << " "
    //           << typeid(T).name() << std::endl;
    return result;
  }
  template <typename T>
  void destroy(T *ptr) {
    if (ptr == nullptr) {
      return;
    }

    if (TRACK_ALLOC) {
      if (current_allocations().find(ptr) == current_allocations().end()) {
        if (allocations().find(ptr) != allocations().end()) {
          PANIC("Double free");
        } else {
          PANIC("Freeing foreign memory");
        }
      }
    }
    // std::cout << "free: " << (size_t)ptr << " "
    //           << std::get<0>(current_allocations()[ptr]) << " "
    //           << std::get<1>(current_allocations()[ptr]) << " "
    //           << std::get<2>(current_allocations()[ptr]) << std::endl;
    if (TRACK_ALLOC) {
      current_allocations().erase(ptr);
    }
    singleton_alloc().destroy(ptr);
  }
  template <typename T>
  T *malloc(size_t size) {
    auto result = singleton_alloc().malloc<T>(size);
    if (TRACK_ALLOC) {
      auto i = id();
      allocations()[result] = std::make_tuple(i, sizeof(T), typeid(T).name());
      current_allocations()[result] =
          std::make_tuple(i, sizeof(T), typeid(T).name());
    }
    // std::cout << "mall: " << (size_t)result << " " << i << " " << sizeof(T)
    // << " "
    //           << typeid(T).name() << std::endl;
    return result;
  }

  static void init() {}
  static int &allocation_count() {
    static int allocation_count = 0;
    return allocation_count;
  }

  static int id() {
    static int next = 0;
    return next++;
  }
  static size_t &max_alloc() {
    static size_t max_alloc = 0;
    return max_alloc;
  }
  static std::unordered_map<void *, std::tuple<int, size_t, std::string>> &
  allocations() {
    static std::unordered_map<void *, std::tuple<int, size_t, std::string>>
        allocations;
    size_t total_size = std::accumulate(allocations.begin(), allocations.end(),
                                        0, [](size_t acc, const auto &kv) {
                                          return acc + std::get<1>(kv.second);
                                        });
    max_alloc() = std::max(max_alloc(), total_size);
    return allocations;
  }
  static std::unordered_map<void *, std::tuple<int, size_t, std::string>> &
  current_allocations() {
    static std::unordered_map<void *, std::tuple<int, size_t, std::string>>
        allocations;
    return allocations;
  }
};

static bool ENABLE_MANAGED_DEALLOC = true;
struct ManagedAlloc {
  static std::unordered_map<void *, std::tuple<size_t, std::string>> &
  allocations() {
    static std::unordered_map<void *, std::tuple<size_t, std::string>>
        allocations;
    return allocations;
  }
  static std::multimap<void *, size_t> &all_allocations() {
    static std::multimap<void *, size_t> all_allocations;
    return all_allocations;
  }

  template <typename T, typename... Args>
  static T *make(Args &&...args) {
    T *ptr;
    cudaCheck(cudaMallocManaged(&ptr, sizeof(T)));
    new (ptr) T(std::forward<Args>(args)...);

    if constexpr (LOG_ALLOC) {
      std::cout << "make:   Allocating ptr " << ptr << " of size " << sizeof(T)
                << " type " << typeid(T).name() << std::endl;
    }

    if constexpr (TRACK_ALLOC) {
      allocations()[ptr] = std::make_tuple(sizeof(T), typeid(T).name());
      all_allocations().insert({ptr, sizeof(T)});
    }

    return ptr;
  }

  template <typename T>
  static void destroy(T *ptr) {
    if (ptr == nullptr) {
      return;
    }
    if (ENABLE_MANAGED_DEALLOC) {
      cudaFree(ptr);

      if constexpr (LOG_ALLOC) {
        std::cout << "destroy: Freeing ptr " << ptr << std::endl;
      }
      if constexpr (TRACK_ALLOC) {
        if (all_allocations().find(ptr) != all_allocations().end() &&
            allocations().find(ptr) == allocations().end()) {
          throw std::runtime_error("ManagedAlloc: double free");
        }
        if (all_allocations().find(ptr) == all_allocations().end()) {
          throw std::runtime_error("ManagedAlloc: freeing foreign memory");
        }
        allocations().erase(ptr);
      }
    }
  }

  template <typename T>
  static T *malloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }

    T *ptr;
    cudaCheck(cudaMallocManaged(&ptr, sizeof(T) * size));

    if constexpr (LOG_ALLOC) {
      std::cout << "malloc: Allocating ptr " << ptr << " of size "
                << sizeof(T) * size << " type " << typeid(T).name()
                << std::endl;
    }
    if constexpr (TRACK_ALLOC) {
      allocations()[ptr] = std::make_tuple(sizeof(T) * size, typeid(T).name());
      all_allocations().insert({ptr, sizeof(T) * size});
    }
    return ptr;
  }
};

template <typename T>
class managed_ptr {
 private:
  T *data;

 public:
  // Explicit constructor
  explicit managed_ptr(T *data) : data(data) {}

  template <typename... Args>
  static managed_ptr make(Args &&...args) {
    auto result =
        managed_ptr(ManagedAlloc::make<T>(std::forward<Args>(args)...));
    cudaCheck(
        cudaMemAdvise(result.data, sizeof(T), cudaMemAdviseSetReadMostly, 0));
    return result;
  }

  managed_ptr() : data(nullptr) {}
  ~managed_ptr() {
    TRACE_START(managed_ptr__destroy);
    if (data) {
      data->~T();
      ManagedAlloc::destroy(data);
    }
  }

  // Constructor/Assignment that binds to nullptr
  // This makes usage with nullptr cleaner
  managed_ptr(std::nullptr_t) : data(nullptr) {}
  managed_ptr &operator=(std::nullptr_t) {
    reset();
    return *this;
  }

  // Constructor/Assignment that allows move semantics
  managed_ptr(managed_ptr &&moving) noexcept : data(nullptr) {
    moving.swap(*this);
  }
  managed_ptr &operator=(managed_ptr &&moving) noexcept {
    moving.swap(*this);
    return *this;
    // See move constructor.
  }

  // Remove compiler generated copy semantics.
  managed_ptr(managed_ptr const &) = delete;
  managed_ptr &operator=(managed_ptr const &) = delete;

  managed_ptr<T> clone() const { return managed_ptr<T>::make(*data); }

  // Const correct access owned object
  T *operator->() const { return data; }
  T &operator*() const { return *data; }

  // Access to smart pointer state
  T *get() const { return data; }
  explicit operator bool() const { return data; }

  // Modify object state
  T *release() noexcept {
    T *result = nullptr;
    std::swap(result, data);
    return result;
  }
  void swap(managed_ptr &src) noexcept { std::swap(data, src.data); }
  void reset() {
    T *tmp = release();
    if (tmp) {
      tmp->~T();
      ManagedAlloc::destroy(tmp);
    }
  }
};
