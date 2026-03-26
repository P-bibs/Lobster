#include "alloc.h"
#include "utils.h"

void *ArenaAlloc::allocate(size_t size) {
  // align to 256 bytes
  auto aligned_size = (size + 255) & ~255;
  void *out = this->current_;
  this->current_ += aligned_size;

  if (this->current_ - this->start_ > this->size_) {
    std::cout << "Used " << current_ - start_ << " bytes ("
              << (float)(current_ - start_) / (float)size_ * 100. << "%) of "
              << size_ << " bytes" << std::endl;
    PANIC("DeviceArenaAllocator: out of memory");
  }
  return out;
}

MAKE_ENV_GETTER(LOG_FROG);

void LeapfrogAlloc::print() {
  std::cout << "\tLeader: \t[" << this->leader_start_ - this->start_ << ", "
            << this->leader_end_ - this->start_ << ")" << std::endl;
  std::cout << "\tFollower:\t[" << this->follower_start_ - this->start_ << ", "
            << this->follower_end_ - this->start_ << ")" << std::endl;
}

void LeapfrogAlloc::new_leader() {
  if (LOG_FROG()) {
    std::cout << "Creating new leader" << std::endl;
    this->print();
  }

  this->follower_start_ = this->leader_start_;
  this->follower_end_ = this->leader_end_;

  this->leader_start_ = this->leader_end_;
  this->leader_end_ = this->leader_start_;
  validate();

  if (LOG_FROG()) {
    std::cout << "New leader created" << std::endl;
    this->print();
  }
}
void LeapfrogAlloc::forget_follower() {
  if (LOG_FROG()) {
    std::cout << "Forgetting follower" << std::endl;
    this->print();
  }

  this->follower_start_ = this->leader_start_;
  this->follower_end_ = this->leader_start_;
  validate();

  if (LOG_FROG()) {
    std::cout << "Follower forgotten" << std::endl;
    this->print();
  }
}
size_t LeapfrogAlloc::follower_size() {
  if (this->follower_start_ < this->follower_end_) {
    return this->follower_end_ - this->follower_start_;
  } else {
    return (this->end() - this->follower_start_) +
           (this->follower_end_ - this->start_);
  }
}
size_t LeapfrogAlloc::leader_size() {
  if (this->leader_start_ < this->leader_end_) {
    return this->leader_end_ - this->leader_start_;
  } else {
    return (this->end() - this->leader_start_) +
           (this->leader_end_ - this->start_);
  }
}
char *LeapfrogAlloc::end() { return this->start_ + this->size_; }

void LeapfrogAlloc::validate() {
  if (!SAFETY) {
    return;
  }
  // in-bounds checks
  if (this->leader_start_ < this->start_) {
    PANIC("Leader start (%p) is %ld bytes behind start (%p)",
          this->leader_start_, this->leader_start_ - this->start_,
          this->start_);
  }
  if (this->leader_start_ > this->end()) {
    PANIC("Leader start (%p) is %ld bytes ahead of end (%p)",
          this->leader_start_, this->leader_start_ - this->end(), this->end());
  }
  if (this->leader_end_ < this->start_) {
    PANIC("Leader end (%p) is %ld bytes behind start (%p)", this->leader_end_,
          this->leader_end_ - this->start_, this->start_);
  }
  if (this->leader_end_ > this->end()) {
    PANIC("Leader end (%p) is %ld bytes ahead of end (%p)", this->leader_end_,
          this->leader_end_ - this->end(), this->end());
  }
  if (this->follower_start_ < this->start_) {
    PANIC("Follower start (%p) is %ld bytes behind start (%p)",
          this->follower_start_, this->follower_start_ - this->start_,
          this->start_);
  }
  if (this->follower_start_ > this->end()) {
    PANIC("Follower start (%p) is %ld bytes ahead of end (%p)",
          this->follower_start_, this->follower_start_ - this->end(),
          this->end());
  }
  if (this->follower_end_ < this->start_) {
    PANIC("Follower end (%p) is %ld bytes behind start (%p)",
          this->follower_end_, this->follower_end_ - this->start_,
          this->start_);
  }
  if (this->follower_end_ > this->end()) {
    PANIC("Follower end (%p) is %ld bytes ahead of end (%p)",
          this->follower_end_, this->follower_end_ - this->end(), this->end());
  }

  // consistency checks
  if ((this->leader_start_ > this->leader_end_) &&
      (this->follower_start_ > this->follower_end_)) {
    std::cout << "Leader and follower have wraparound" << std::endl;
    this->print();
    PANIC("Both leader and follower have wraparound");
  }
}

void *LeapfrogAlloc::allocate(size_t size) {
  // align to 256 bytes
  auto aligned_size = (size + 255) & ~255;

  if (LOG_FROG()) {
    std::cout << "Frog: allocating " << aligned_size << " bytes" << std::endl;
    this->print();
  }

  auto follower_is_behind = this->follower_start_ <= this->leader_end_;
  if (follower_is_behind) {
    auto space_is_available_in_alloc =
        this->leader_end_ + aligned_size < this->end();
    if (space_is_available_in_alloc) {
      void *out = this->leader_end_;
      this->leader_end_ += aligned_size;
      validate();
      return out;
    } else {
      if (LOG_FROG()) {
        std::cout << "Not enough space in alloc, overflowing" << std::endl;
      }
      this->leader_end_ = this->start_;
    }
  }

  auto space_is_available_before_follower =
      this->leader_end_ + aligned_size < this->follower_start_;
  if (space_is_available_before_follower) {
    void *out = this->leader_end_;
    this->leader_end_ += aligned_size;
    validate();
    return out;
  } else {
    PANIC("LeapfrogAlloc: out of memory");
  }
}

bool use_frog() {
  static std::optional<bool> use_frog;
  if (!use_frog.has_value()) {
    auto env = std::getenv("USE_FROG");
    if (env) {
      use_frog = true;
    } else {
      use_frog = false;
    }
  }
  return use_frog.value();
}
Allocator DeviceAlloc::singleton_alloc() {
  static std::optional<Allocator> alloc;
  if (!alloc.has_value()) {
    if (use_frog()) {
      size_t free_mem, total_mem;
      cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
      total_mem *= 0.9;
      size_t arena_size =
          std::getenv("ARENA_SIZE") ? std::stoi(std::getenv("ARENA_SIZE")) * 1000 * 1000 * 1000 : total_mem;
      LeapfrogAlloc *arena = new LeapfrogAlloc(arena_size);
      alloc = Allocator(arena);
    } else {
      alloc = Allocator();
    }
  }
  assert(alloc.has_value());
  return alloc.value();
}

Allocator &lobster_global_allocator() {
  static Allocator alloc = Allocator(DefaultAlloc());
  return alloc;
}

size_t &iter_allocs() {
  static size_t n = 0;
  return n;
}
