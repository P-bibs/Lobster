#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cstddef>
#include <cwchar>
#include <iostream>
#include <iterator>
#include <vector>

#include "alloc.h"
#include "bindings.h"
#include "flame.h"
#include "utils.h"

template <typename T>
class device_vec {
 private:
  size_t length_;
  size_t capacity_;
  Allocator alloc_;
  T *data_;

 public:
  class device_vec_iter {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

   private:
    pointer ptr_;

   public:
    __host__ __device__ device_vec_iter(pointer ptr) : ptr_(ptr) {}
    __device__ reference operator*() const { return *ptr_; }
    __device__ pointer operator->() const { return ptr_; }
    __host__ __device__ device_vec_iter &operator++() {
      ptr_++;
      return *this;
    }
    __host__ __device__ device_vec_iter operator++(int) {
      device_vec_iter tmp = *this;
      ++(*this);
      return tmp;
    }
    __host__ __device__ friend bool operator==(const device_vec_iter &a,
                                               const device_vec_iter &b) {
      return a.ptr_ == b.ptr_;
    }
    __host__ __device__ friend bool operator!=(const device_vec_iter &a,
                                               const device_vec_iter &b) {
      return a.ptr_ != b.ptr_;
    }

    __host__ __device__ device_vec_iter operator+(difference_type n) const {
      return ptr_ + n;
    }
    __host__ __device__ device_vec_iter &operator+=(difference_type n) {
      ptr_ += n;
      return *this;
    }

    __host__ __device__ difference_type
    operator-(const device_vec_iter &other) const {
      return ptr_ - other.ptr_;
    }

    __host__ __device__ device_vec_iter &operator--() {
      ptr_--;
      return *this;
    }
    __host__ __device__ device_vec_iter operator-(int n) const {
      return ptr_ - n;
    }

    __device__ reference operator[](std::size_t index) const {
      return ptr_[index];
    }
  };

  using iterator = device_vec_iter;

  device_vec() : length_(0), capacity_(0), data_(nullptr) {}
  device_vec(const device_vec &other) = delete;
  device_vec operator=(const device_vec &other) = delete;
  device_vec(device_vec &&other) {
    length_ = other.length_;
    capacity_ = other.capacity_;
    alloc_ = other.alloc_;
    data_ = other.data_;
    other.length_ = 0;
    other.capacity_ = 0;
    other.alloc_ = Allocator();
    other.data_ = nullptr;
  }
  device_vec &operator=(device_vec &&other) {
    alloc_.destroy(data_);
    length_ = other.length_;
    capacity_ = other.capacity_;
    alloc_ = other.alloc_;
    data_ = other.data_;
    other.length_ = 0;
    other.capacity_ = 0;
    other.alloc_ = Allocator();
    other.data_ = nullptr;
    return *this;
  }
  ~device_vec() {
    clear();
    alloc_.destroy(data_);
  }

  device_vec(const std::vector<T> &host_vec)
      : length_(host_vec.size()), capacity_(host_vec.size()) {
    TRACE_START(device_vec_alloc);
    data_ = alloc_.malloc<T>(length_);
    cudaCheck(cudaMemcpy(data_, host_vec.data(), length_ * sizeof(T),
                         cudaMemcpyHostToDevice));
  }
  device_vec(const Array<T> &host_vec)
      : length_(host_vec.size()), capacity_(host_vec.size()) {
    TRACE_START(device_vec_alloc);
    data_ = alloc_.malloc<T>(length_);
    cudaCheck(cudaMemcpy(data_, host_vec.data(), length_ * sizeof(T),
                         cudaMemcpyHostToDevice));
  }
  device_vec(size_t length) : length_(length), capacity_(length) {
    TRACE_START(device_vec_alloc);
    data_ = alloc_.malloc<T>(length_);
  }

  device_vec(size_t length, const Allocator &alloc) : length_(length), capacity_(length), alloc_(alloc) {
    TRACE_START(device_vec_alloc);
    data_ = alloc_.malloc<T>(length_);
  }

  explicit device_vec(const thrust::device_vector<T> &device_vec)
      : length_(device_vec.size()), capacity_(device_vec.size()) {
    TRACE_START(device_vec_alloc);
    data_ = alloc_.malloc<T>(length_);
    thrust::copy(device_vec.begin(), device_vec.end(), data_);
  }

  std::vector<T> to_host() const {
    std::vector<T> host_vec(length_);
    cudaCheck(cudaMemcpy(host_vec.data(), data_, length_ * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return host_vec;
  }

  device_vec<T> to(const Allocator &alloc) const {
    device_vec<T> new_vec;
    new_vec.alloc_ = alloc;
    new_vec.reserve(capacity_);
    cudaCheck(cudaMemcpy(new_vec.data_, data_, length_ * sizeof(T),
                         cudaMemcpyDeviceToDevice));
    new_vec.length_ = length_;
    return new_vec;
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      T *new_data = alloc_.malloc<T>(new_capacity);
      cudaCheck(cudaMemcpy(new_data, data_, length_ * sizeof(T),
                           cudaMemcpyDeviceToDevice));
      alloc_.destroy(data_);
      data_ = new_data;
      capacity_ = new_capacity;
    }
  }

  void resize(size_t new_length) {
    if (new_length > capacity_) {
      reserve(new_length);
    }
    length_ = new_length;
  }

  void clear() {
    //for (size_t i = 0; i < length_; i++) {
    //  data_[i].~T();
    //}
    length_ = 0;
  }

  __host__ device_vec clone() const {
    return this->to(alloc_);
  }

  void push_back(const T &value) {
    if (length_ == capacity_) {
      reserve(capacity_ * 2 + 1);
    }
    cudaCheck(
        cudaMemcpy(data_ + length_, &value, sizeof(T), cudaMemcpyHostToDevice));
    length_++;
  }

  void append(const device_vec &other) {
    if (length_ + other.length_ > capacity_) {
      reserve(length_ + other.length_);
    }
    cudaCheck(cudaMemcpy(data_ + length_, other.data_,
                         other.length_ * sizeof(T), cudaMemcpyDeviceToDevice));
    length_ += other.length_;
  }

  __host__ T at_host(size_t i) const {
    if (SAFETY) {
      if (i >= length_) {
        PANIC("device_vec host_at: index out of bounds");
      }
    }
    T t;
    cudaCheck(cudaMemcpy(&t, data_ + i, sizeof(T), cudaMemcpyDeviceToHost));
    return t;
  }
  __host__ void set_from_host(size_t i, const T &value) {
    if (SAFETY) {
      if (i >= length_) {
        PANIC("device_vec set_from_host: index out of bounds");
      }
    }
    cudaCheck(cudaMemcpy(data_ + i, &value, sizeof(T), cudaMemcpyHostToDevice));
  }
  __device__ T &at_device(size_t i) {
    if (SAFETY) {
      if (i >= length_) {
        PANIC("device_vec device_at: index %lu out of bounds for length %lu", i, length_);
      }
    }
    return data_[i];
  }
  __device__ const T &at_device(size_t i) const {
    if (SAFETY) {
      if (i >= length_) {
        PANIC("device_vec device_at: index %lu out of bounds for length %lu", i, length_);
      }
    }
    return data_[i];
  }

  T at_host(size_t i) {
    if (SAFETY) {
      if (i >= length_) {
        PANIC("device_vec host_at: index out of bounds");
      }
    }

    T t;
    cudaCheck(cudaMemcpy(&t, data_ + i, sizeof(T), cudaMemcpyDeviceToHost));
    return t;
  }

  device_vec_iter begin() { return device_vec_iter(data_); }
  device_vec_iter end() { return device_vec_iter(data_ + length_); }
  const device_vec_iter cbegin() const { return device_vec_iter(data_); }
  const device_vec_iter cend() const {
    return device_vec_iter(data_ + length_);
  }
  __host__ __device__ T *data() const { return data_; }

  __host__ __device__ size_t size() const { return length_; }
  __host__ __device__ size_t capacity() const { return capacity_; }

  friend std::ostream &operator<<(std::ostream &os, const device_vec &vec) {
    auto host_vec = vec.to_host();
    os << "[";
    for (size_t i = 0; i < host_vec.size(); i++) {
      os << host_vec[i];
      if (i < host_vec.size() - 1) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }
};

struct TableHostCopierFunctor {
  void *source;
  Value *dest;
  Value::Tag tag;
  uint32_t stride;

  TableHostCopierFunctor(void *source, Value *dest, Value::Tag tag, uint32_t stride)
      : source(source), dest(dest), tag(tag), stride(stride) {}

  __device__ void operator()(size_t row) {
    // memcpy k bytes from source[row] to dest[row]
    for (size_t k = 0; k < stride; k++) {
      reinterpret_cast<char *>(&dest[row].i8)[k] =
          (reinterpret_cast<const char *>(source) + row * stride)[k];
    }
    // copy the tag to the destination
    dest[row].tag = tag;
  }
};

template <int N>
__global__ void TableHostCopier(void *source, Value *dest, Value::Tag tag, uint32_t stride, size_t length) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < length) {
    // memcpy k bytes from source[row] to dest[row]
    for (size_t k = 0; k < stride; k++) {
      char byte = (reinterpret_cast<const char *>(source) + row * stride)[k];
      reinterpret_cast<char *>(&dest[row].i8)[k] = byte;
    }
    // copy the tag to the destination
    dest[row].tag = tag;
  }
}

template <typename T>
DeviceArray<T *> device_view(Array<device_vec<T>&> &vecs) {
  Array<T *> result(vecs.size());
  for (size_t i = 0; i < vecs.size(); i++) {
    result[i] = vecs[i].data();
  }
  return result.to();
}
template <typename T>
DeviceArray<T *> device_view(Array<device_vec<T>> &vecs) {
  Array<T *> result(vecs.size());
  for (size_t i = 0; i < vecs.size(); i++) {
    result[i] = vecs[i].data();
  }
  return result.to();
}




class device_buffer {
 private:
  size_t length_;
  size_t capacity_;
  size_t stride_;
  ValueType type_;
  Allocator alloc_;
  void *data_;

 public:
  template <typename T>
  class device_buffer_iter {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

   private:
    pointer ptr_;

   public:
    __host__ __device__ device_buffer_iter() : ptr_(nullptr) {}
    __host__ __device__ device_buffer_iter(pointer ptr) : ptr_(ptr) {}
    __device__ reference operator*() const { return *ptr_; }
    __device__ pointer operator->() const { return ptr_; }
    __host__ __device__ device_buffer_iter &operator++() {
      ptr_++;
      return *this;
    }
    __host__ __device__ device_buffer_iter operator++(int) {
      device_buffer_iter tmp = *this;
      ++(*this);
      return tmp;
    }
    __host__ __device__ friend bool operator==(const device_buffer_iter &a,
                                               const device_buffer_iter &b) {
      return a.ptr_ == b.ptr_;
    }
    __host__ __device__ friend bool operator!=(const device_buffer_iter &a,
                                               const device_buffer_iter &b) {
      return a.ptr_ != b.ptr_;
    }

    __host__ __device__ device_buffer_iter operator+(difference_type n) const {
      return ptr_ + n;
    }

    __host__ __device__ difference_type
    operator-(const device_buffer_iter &other) const {
      return ptr_ - other.ptr_;
    }
    __host__ __device__ device_buffer_iter &operator+=(difference_type n) {
      ptr_ += n;
      return *this;
    }
    __host__ __device__ device_buffer_iter &operator--() {
      ptr_--;
      return *this;
    }

    __device__ reference operator[](std::size_t index) const {
      return ptr_[index];
    }

    __device__ __host__ pointer data() const { return ptr_; }
  };

  template <typename T>
  using iterator = device_buffer_iter<T>;

  device_buffer()
      : length_(0), capacity_(0), stride_(1), type_(), data_(nullptr) {}
  device_buffer(ValueType type)
      : length_(0),
        capacity_(0),
        stride_(type.size()),
        type_(type),
        data_(nullptr) {}

  device_buffer(size_t length, ValueType type)
      : length_(length), capacity_(length), stride_(type.size()), type_(type) {
    data_ = alloc_.malloc<char>(length_ * stride_);
  }
  device_buffer(size_t length, ValueType type, const Allocator &alloc)
      : length_(length), capacity_(length), stride_(type.size()), type_(type), alloc_(alloc) {
    data_ = alloc_.malloc<char>(length_ * stride_);
  }
  device_buffer(const device_buffer &other) = delete;
  device_buffer operator=(const device_buffer &other) = delete;
  device_buffer(device_buffer &&other) {
    length_ = other.length_;
    capacity_ = other.capacity_;
    stride_ = other.stride_;
    type_ = other.type_;
    alloc_ = other.alloc_;
    data_ = other.data_;
    other.length_ = 0;
    other.capacity_ = 0;
    other.stride_ = 1;
    other.alloc_ = Allocator();
    other.data_ = nullptr;
  }
  device_buffer &operator=(device_buffer &&other) {
    length_ = other.length_;
    capacity_ = other.capacity_;
    stride_ = other.stride_;
    type_ = other.type_;
    alloc_ = other.alloc_;
    data_ = other.data_;
    other.length_ = 0;
    other.capacity_ = 0;
    other.stride_ = 1;
    other.alloc_ = Allocator();
    other.data_ = nullptr;
    return *this;
  }
  ~device_buffer() {
    clear();
    alloc_.destroy((char*)data_);
  }

  device_buffer(const std::vector<Value> &host_vec, ValueType type)
      : length_(host_vec.size()),
        capacity_(host_vec.size()),
        stride_(type.size()),
        type_(type) {
    if (host_vec.size() > 0) {
      assert(stride_ == host_vec[0].type().size());
      data_ = alloc_.malloc<char>(length_ * stride_);

      DISPATCH_ON_TYPE(
          type_, T, std::vector<T> buf; for (auto &v
                                             : host_vec) {
            buf.push_back(*reinterpret_cast<const T *>(&v.i8._0));
          } cudaMemcpy(data_, buf.data(), length_ * stride_,
                       cudaMemcpyHostToDevice);
          cudaCheck(cudaDeviceSynchronize()););
    } else {
      data_ = nullptr;
    }
  }
  device_buffer(const Array<Value> &host_vec, ValueType type)
      : length_(host_vec.size()),
        capacity_(host_vec.size()),
        stride_(type.size()),
        type_(type) {
    if (host_vec.size() > 0) {
      assert(stride_ == host_vec[0].type().size());
      data_ = alloc_.malloc<char>(length_ * stride_);

      DISPATCH_ON_TYPE(
          type_, T, std::vector<T> buf;
          for (size_t i = 0; i < host_vec.size(); i++) {
            auto &v = host_vec[i];
            buf.push_back(*reinterpret_cast<const T *>(&v.i8._0));
          } cudaMemcpy(data_, buf.data(), length_ * stride_,
                       cudaMemcpyHostToDevice);
          cudaCheck(cudaDeviceSynchronize()););
    } else {
      data_ = nullptr;
    }
  }

  template <typename T>
  std::vector<T> to_host() const {
    std::vector<T> host_vec(length_);
    cudaCheck(cudaMemcpy(host_vec.data(), data_, length_ * stride_,
                         cudaMemcpyDeviceToHost));
    return host_vec;
  }

  Array<Value> to_host_tagged(Value::Tag tag) const {
    if (length_ == 0) {
      return Array<Value>();
    }
    device_vec<Value> values(length_);
    TableHostCopier<1>
      <<<ROUND_UP_TO_NEAREST(length_, 512), 512>>>(
          data(), values.data(), tag, stride_, length_);
    cudaCheck(cudaDeviceSynchronize());

    Array<Value> values_host(length_);

    cudaCheck(cudaMemcpy(values_host.data(), values.data(),
                         length_ * sizeof(Value), cudaMemcpyDeviceToHost));

    return values_host;
  }

  device_buffer to(const Allocator &alloc) const {
    device_buffer new_buffer(type_);
    new_buffer.alloc_ = alloc;
    new_buffer.reserve(capacity_);
    cudaCheck(cudaMemcpy(new_buffer.data_, data_, length_ * stride_,
                         cudaMemcpyDeviceToDevice));
    new_buffer.length_ = length_;
    return new_buffer;
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      void *new_data = alloc_.malloc<char>(new_capacity * stride_);
      cudaMemcpy(new_data, data_, length_ * stride_, cudaMemcpyDeviceToDevice);
      alloc_.destroy((char*)data_);
      data_ = new_data;
      capacity_ = new_capacity;
    }
  }

  void resize(size_t new_length) {
    if (new_length > capacity_) {
      reserve(new_length);
    }
    length_ = new_length;
  }

  void clear() { length_ = 0; }

  __host__ device_buffer clone() const {
    device_buffer new_vec(type_);
    new_vec.reserve(capacity_);
    cudaCheck(cudaMemcpy(new_vec.data_, data_, length_ * stride_,
                         cudaMemcpyDeviceToDevice));
    new_vec.length_ = length_;
    return new_vec;
  }

  void append(const device_buffer &other) {
    if (SAFETY) {
      if (type_ != other.type_) {
        throw std::runtime_error("device_buffer append: types do not match");
      }
      if (stride_ != other.stride_) {
        throw std::runtime_error("device_buffer append: strides do not match");
      }
    }
    if (length_ + other.length_ > capacity_) {
      reserve(length_ + other.length_);
    }
    cudaCheck(cudaMemcpy(static_cast<char *>(data_) + length_ * stride_,
                         other.data_, other.length_ * stride_,
                         cudaMemcpyDeviceToDevice));
    length_ += other.length_;
  }

  template <typename T>
  __device__ T &at(size_t i) {
    if (SAFETY) {
      if (i >= length_) {
        PANIC(
            "device_buffer device_at: index out of bounds (index %zu, "
            "length %zu)",
            i, length_);
      }
    }
    return static_cast<T *>(data_)[i];
  }
  template <typename T>
  __device__ const T &at(size_t i) const {
    if (SAFETY) {
      if (i >= length_) {
        PANIC(
            "device_buffer device_at: index out of bounds (index %zu, "
            "length %zu)",
            i, length_);
      }
    }
    return static_cast<T *>(data_)[i];
  }

  __host__ __device__ const void *at_raw(size_t i) const {
    if (SAFETY) {
      if (i >= length_) {
        PANIC(
            "device_buffer device_at: index out of bounds (index %zu, "
            "length %zu)",
            i, length_);
      }
    }
    return static_cast<char *>(data_) + i * stride_;
  }

  template <typename T>
  device_buffer_iter<T> begin() {
    return device_buffer_iter<T>(static_cast<T *>(data_));
  }
  template <typename T>
  device_buffer_iter<T> end() {
    return device_buffer_iter<T>(static_cast<T *>(data_)) + length_;
  }
  template <typename T>
  const device_buffer_iter<T> cbegin() const {
    return device_buffer_iter<T>(static_cast<T *>(data_));
  }
  template <typename T>
  const device_buffer_iter<T> cend() const {
    return device_buffer_iter<T>(static_cast<T *>(data_)) + length_;
  }
  //template <typename T>
  //T *begin() {
  //  return static_cast<T *>(data_);
  //}
  //template <typename T>
  //T *end() {
  //  return static_cast<T *>(data_) + length_;
  //}
  //template <typename T>
  //const T *cbegin() const {
  //  return static_cast<T *>(data_);
  //}
  //template <typename T>
  //const T *cend() const {
  //  return static_cast<T *>(data_) + length_;
  //}

  __host__ __device__ size_t size() const { return length_; }
  __host__ __device__ size_t capacity() const { return capacity_; }
  __host__ __device__ size_t stride() const { return stride_; }
  __host__ __device__ ValueType type() const { return type_; }
  __host__ __device__ void *data() const { return data_; }

  friend std::ostream &operator<<(std::ostream &os, const device_buffer &vec) {
    DISPATCH_ON_TYPE(
        vec.type(), T, auto host_vec = vec.to_host<T>();
        os << "["; for (size_t i = 0; i < host_vec.size(); i++) {
          os << +host_vec[i];
          if (i < host_vec.size() - 1) {
            os << ", ";
          }
        } os << "]";
        return os;);
  }
};
