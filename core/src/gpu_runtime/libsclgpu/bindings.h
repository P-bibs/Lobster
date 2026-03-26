#pragma once

#include <thrust/copy.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <functional>

#include "alloc.h"
#include "utils.h"
#include "flame.h"

constexpr static const uint64_t DEFAULT_RANDOM_SEED = 1234;

constexpr static const uintptr_t TUPLE_ACCESSOR_DEPTH = 3;

class ValueType {
 public:
  enum class Tag {
    // TODO: add this `None` type to Rust
    None,

    I8,
    I16,
    I32,
    I64,
    ISize,
    U8,
    U16,
    U32,
    U64,
    USize,
    F32,
    F64,
    Char,
    Bool,
    Str,
    String,
    Symbol,
    Entity,
  };
  Tag tag_;

  __host__ __device__ ValueType();
  __host__ __device__ constexpr ValueType(Tag tag) : tag_(tag) {}
  __host__ __device__ Tag tag() const { return this->tag_; }

  __host__ __device__ bool operator==(const ValueType &other) const;
  __host__ __device__ bool operator!=(const ValueType &other) const;
  __host__ __device__ size_t size() const;

  __host__ __device__ static constexpr ValueType None() {
    return ValueType(Tag::None);
  }
  __host__ __device__ static constexpr ValueType I8() {
    return ValueType(Tag::I8);
  }
  __host__ __device__ static constexpr ValueType I16() {
    return ValueType(Tag::I16);
  }
  __host__ __device__ static constexpr ValueType I32() {
    return ValueType(Tag::I32);
  }
  __host__ __device__ static constexpr ValueType I64() {
    return ValueType(Tag::I64);
  }
  __host__ __device__ static constexpr ValueType ISize() {
    return ValueType(Tag::ISize);
  }
  __host__ __device__ static constexpr ValueType U8() {
    return ValueType(Tag::U8);
  }
  __host__ __device__ static constexpr ValueType U16() {
    return ValueType(Tag::U16);
  }
  __host__ __device__ static constexpr ValueType U32() {
    return ValueType(Tag::U32);
  }
  __host__ __device__ static constexpr ValueType U64() {
    return ValueType(Tag::U64);
  }
  __host__ __device__ static constexpr ValueType USize() {
    return ValueType(Tag::USize);
  }
  __host__ __device__ static constexpr ValueType F32() {
    return ValueType(Tag::F32);
  }
  __host__ __device__ static constexpr ValueType F64() {
    return ValueType(Tag::F64);
  }
  __host__ __device__ static constexpr ValueType Char() {
    return ValueType(Tag::Char);
  }
  __host__ __device__ static constexpr ValueType Bool() {
    return ValueType(Tag::Bool);
  }
  __host__ __device__ static constexpr ValueType Str() {
    return ValueType(Tag::Str);
  }
  __host__ __device__ static constexpr ValueType String() {
    return ValueType(Tag::String);
  }
  __host__ __device__ static constexpr ValueType Symbol() {
    return ValueType(Tag::Symbol);
  }
  __host__ __device__ static constexpr ValueType Entity() {
    return ValueType(Tag::Entity);
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const ValueType &value_type);
};

struct ValueU32 {
  using type = uint32_t;
  using sort_type = type;
  __device__ static bool print(type value) { printf("%u", value); return true; }
  static constexpr ValueType tag_ = ValueType::U32();
  static constexpr ValueType tag() { return tag_; }
};
struct ValueUSize {
  using type = uintptr_t;
  using sort_type = type;
  __device__ static bool print(type value) { printf("%lu", value); return true; }
  static constexpr ValueType tag_ = ValueType::USize();
  static constexpr ValueType tag() { return tag_; }
};
struct ValueF32 {
  using type = float;
  using sort_type = uint32_t;
  __device__ static bool print(type value) { printf("%f", value); return true; }
  static constexpr ValueType tag_ = ValueType::F32();
  static constexpr ValueType tag() { return tag_; }
};
struct ValueSymbol {
  using type = uintptr_t;
  using sort_type = type;
  __device__ static bool print(type value) { printf("%lu", value); return true; }
  static constexpr ValueType tag_ = ValueType::Symbol();
  static constexpr ValueType tag() { return tag_; }
};

class String {
 private:
  uintptr_t length_;
  uint8_t *value_;

 public:
  String(uintptr_t length, uint8_t *value);
  String(const String &string);
  String &operator=(const String &other);
  ~String();

  std::string to_string() const;

  uintptr_t length() const;
  const uint8_t *data() const;

  friend std::ostream &operator<<(std::ostream &os, const String &string);
};

template <typename T, typename Alloc = HostAlloc>
class Array {
 private:
  size_t length_;
  T *values_;

 public:
  void validate() {
    if (!SAFETY) {
      return;
    }
    if (length_ != 0) {
      if (values_ == nullptr) {
        PANIC("values_ is null");
      }
    }
    if (values_ == nullptr) {
      if (length_ != 0) {
        PANIC("values_ is null");
      }
    }
  }
  __host__ __device__ Array() : length_(0), values_(nullptr) {}
  Array(size_t length, const T *values) {
    length_ = length;
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < length_; i++) {
      new (temp_values + i) T(values[i]);
    }
    this->values_ = temp_values;
  }
  explicit Array(size_t length) : length_(length) {
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < length_; i++) {
      if constexpr (std::is_same_v<Alloc, HostAlloc> && std::is_default_constructible_v<T>) {
        new (temp_values + i) T();
      }
    }
    this->values_ = temp_values;
    validate();
  }
  /* Array(size_t length, std::function<T(size_t)> generator) { */
  /*   length_ = length; */
  /*   auto temp_values = (void *)malloc(sizeof(T) * length_); */
  /*   for (size_t i = 0; i < length_; i++) { */
  /*     new ((T *)temp_values + i) T(generator(i)); */
  /*   } */
  /*   this->values_ = (T *)temp_values; */
  /* } */
  explicit Array(const T &value) : length_(1) {
    values_ = Alloc::template malloc<T>(1);
    new (values_) T(value);
  }
  Array(std::vector<T> &values) : length_(values.size()) {
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < length_; i++) {
      new (temp_values + i) T(values[i]);
    }
    this->values_ = temp_values;
  }
  Array(std::vector<Array<T, Alloc>> &values) {
    size_t length = 0;
    for (auto &value : values) {
      length += value.size();
    }
    auto temp_values = Alloc::template malloc<T>(length);
    size_t index = 0;
    for (auto &value : values) {
      for (size_t i = 0; i < value.size(); i++) {
        new (temp_values + index + i) T(value[i]);
      }
      index += value.size();
    }
    this->length_ = length;
    this->values_ = temp_values;
  }
  Array(Array<T, Alloc> &&other) {
    this->length_ = other.length_;
    this->values_ = other.values_;
    other.length_ = 0;
    other.values_ = nullptr;
  }
  Array &operator=(Array<T, Alloc> &&other) {
    clear();
    this->length_ = other.length_;
    this->values_ = other.values_;
    other.length_ = 0;
    other.values_ = nullptr;
    return *this;
  }

  Array(const Array<T, Alloc> &array, const T &value)
      : length_(array.length_ + 1) {
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < array.length_; i++) {
      new (temp_values + i) T(array.values_[i]);
    }
    new (temp_values + array.length_) T(value);
    this->values_ = temp_values;
  }
  Array(const T &value, const Array<T, Alloc> &array)
      : length_(array.length_ + 1) {
    auto temp_values = Alloc::template malloc<T>(length_);
    new (temp_values) T(value);
    for (size_t i = 0; i < array.length_; i++) {
      new (temp_values + i + 1) T(array.values_[i]);
    }
    this->values_ = temp_values;
  }
  Array(const Array<T, Alloc> &array1, const Array<T, Alloc> &array2)
      : length_(array1.length_ + array2.length_) {
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < array1.length_; i++) {
      new (temp_values + i) T(array1.values_[i]);
    }
    for (size_t i = 0; i < array2.length_; i++) {
      new (temp_values + i + array1.length_) T(array2.values_[i]);
    }
    this->values_ = temp_values;
  }

  Array(const Array<T, Alloc> &array) : length_(array.length_) {
    auto temp_values = Alloc::template malloc<T>(length_);
    for (size_t i = 0; i < length_; i++) {
      new (temp_values + i) T(array.values_[i]);
    }
    this->values_ = temp_values;
    validate();
  }

  void clear() {
    validate();
    for (size_t i = 0; i < this->length_; i++) {
      this->values_[i].~T();
    }
    Alloc::template destroy<T>(this->values_);
    this->length_ = 0;
    this->values_ = nullptr;
  }
  ~Array() { clear(); }

  bool operator==(const Array<T, Alloc> &other) const {
    if (this->length_ != other.length_) {
      return false;
    }
    for (size_t i = 0; i < this->length_; i++) {
      if (this->values_[i] != other.values_[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename NewAlloc = Alloc>
  Array<T, NewAlloc> clone() const {
    auto temp_values = NewAlloc::template malloc<T>(length_);
    for (size_t i = 0; i < length_; i++) {
      new (temp_values + i) T(values_[i].clone());
    }
    Array new_arr;
    new_arr.length_ = length_;
    new_arr.values_ = temp_values;
    return new_arr;
  }

  Array<T, DeviceAlloc> to() const {
    if constexpr (!std::is_same_v<Alloc, HostAlloc>) {
      PANIC("to() only works for converting DeviceArrays to HostArrays");
    }
    Array<T, DeviceAlloc> output(this->length_);
    cudaCheck(cudaMemcpy(output.data(), this->values_, length_ * sizeof(T),
                         cudaMemcpyHostToDevice));
    return std::move(output);
  }

  __host__ __device__ size_t size() const { return this->length_; }
  __host__ __device__ T *data() const { return this->values_; }

  __host__ __device__ const T &operator[](size_t index) const {
    if (SAFETY) {
      if (index >= this->size()) {
        PANIC("index out of range");
      }
    }
    return this->values_[index];
  }

  __host__ __device__ T &operator[](size_t index) {
    if (SAFETY) {
      if (index >= this->size()) {
        PANIC("index out of range");
      }
    }
    return this->values_[index];
  }

  friend std::ostream &operator<<(std::ostream &os, const Array &array) {
    os << "[";
    for (size_t i = 0; i < array.size(); i++) {
      os << array[i];
      if (i < array.size() - 1) {
        os << ", ";
      }
    }
    return os << "]";
  }
};
template <typename T>
using HostArray = Array<T, HostAlloc>;
template <typename T>
using ManagedArray = Array<T, ManagedAlloc>;
template <typename T>
using DeviceArray = Array<T, DeviceAlloc>;

class ForeignTupleType {
 public:
  enum class Tag {
    Tuple,
    Unit,
  };

  struct Tuple_Body {
    Array<ForeignTupleType> _0;
  };

  struct Unit_Body {
    ValueType _0;
  };

  Tag tag;
  union {
    Tuple_Body tuple;
    Unit_Body unit;
  };

  ForeignTupleType(Array<ForeignTupleType> tuple);
  ForeignTupleType(std::vector<ForeignTupleType> tuple);
  ForeignTupleType(ValueType unit);

  ForeignTupleType(const ForeignTupleType &tuple_type);
  ForeignTupleType &operator=(const ForeignTupleType &other);
  ~ForeignTupleType();

  ForeignTupleType clone() const;

  bool operator==(const ForeignTupleType &other) const;
  bool operator!=(const ForeignTupleType &other) const;
  size_t width() const;

  std::pair<ForeignTupleType, Array<ForeignTupleType>> remove_first() const;

  ForeignTupleType append(const ForeignTupleType &other) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const ForeignTupleType &tuple_type);
};

class TupleType {
 private:
  static constexpr uint8_t SIZE = 14;
  // the number of values in the tuple
  uint8_t width_;
  // the size of the data_ array
  uint8_t valid_bytes_;
  // a representation of the tuple where each value is either OPEN_PAREN,
  // CLOSE_PAREN, or a ValueType
  uint8_t data_[SIZE];

  static const uint8_t OPEN_PAREN = 0b11111111;
  static const uint8_t CLOSE_PAREN = 0b01111111;

  enum class Kind { OpenParen, CloseParen, Type };

  __host__ __device__ TupleType();
  __host__ __device__ TupleType(const TupleType &other, size_t start);
  __host__ __device__ static Kind get_kind(uint8_t byte);
  __host__ __device__ uint8_t find_close_paren(size_t start) const;
  __host__ __device__ uint8_t valid_bytes() const;
  __host__ __device__ size_t index_to_location(size_t index) const;

 public:
  __host__ __device__ void validate();
  TupleType(ForeignTupleType tuple);
  __host__ __device__ TupleType(ValueType value);
  TupleType(Array<TupleType> tuple);
  TupleType(std::vector<TupleType> tuple);
  TupleType(const TupleType &tuple_type) = default;
  TupleType &operator=(const TupleType &other) = default;

  bool operator==(const TupleType &other) const;
  bool operator!=(const TupleType &other) const;

  __host__ __device__ size_t width() const;

  __host__ __device__ size_t size() const;

  __host__ __device__ size_t bytewidth() const;

  __host__ __device__ bool is_nested() const;
  __host__ __device__ bool is_scalar() const;

  __host__ __device__ ValueType singleton();

  std::pair<TupleType, TupleType> remove_first() const;
  TupleType append(const TupleType &other) const;
  TupleType flatten() const;
  std::vector<ValueType> to_flat_vector() const;

  ForeignTupleType to_foreign() const;

  __host__ __device__ TupleType at(size_t index) const;

  size_t accessor_to_index(std::vector<int8_t> indices) const;
  ValueType accessor_to_type(std::vector<int8_t> indices) const;
  friend std::ostream &operator<<(std::ostream &os,
                                  const TupleType &tuple_type);
};
static_assert(sizeof(TupleType) <= 32, "TupleType too large");

template <typename T>
__device__ __host__ uint32_t simple_hash(T key) {
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key;
}
template <>
inline __device__ __host__ uint32_t simple_hash(float key) {
  return simple_hash(*(uint32_t *)&key);
}
template <>
inline __device__ __host__ uint32_t simple_hash(double key) {
  return simple_hash(*(uint64_t *)&key);
}

#ifdef NON_COMMUTATIVE_HASH
template <typename T>
__device__ __host__ uint32_t multi_hash(T key1) {
  return simple_hash(key1);
}

template <typename T, typename U>
__device__ __host__ uint32_t multi_hash(T key1, U key2) {
  auto hash1 = simple_hash(key1);
  auto hash2 = simple_hash(key2);
  return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash2 >> 2));
}
template <typename T, typename U, typename V>
__device__ __host__ uint32_t multi_hash(T key1, U key2, V key3) {
  return multi_hash(key1, multi_hash(key2, key3));
}

template <typename T, typename U, typename V, typename W>
__device__ __host__ uint32_t multi_hash(T key1, U key2, V key3, W key4) {
  return multi_hash(key1, multi_hash(key2, key3, key4));
}
#else
template <typename... T>
__device__ __host__ uint32_t multi_hash(T... keys) {
  return (... + simple_hash(keys));
}
#endif

class Value {
 public:
  enum class Tag {
    None,

    I8,
    I16,
    I32,
    I64,
    ISize,
    U8,
    U16,
    U32,
    U64,
    USize,
    F32,
    F64,
    Char,
    Bool,
    String,
    Symbol,
    Entity,
  };

  static Tag tag_from_type(ValueType::Tag t);

  struct I8_Body {
    int8_t _0;
  };

  struct I16_Body {
    int16_t _0;
  };

  struct I32_Body {
    int32_t _0;
  };

  struct I64_Body {
    int64_t _0;
  };

  struct ISize_Body {
    intptr_t _0;
  };

  struct U8_Body {
    uint8_t _0;
  };

  struct U16_Body {
    uint16_t _0;
  };

  struct U32_Body {
    uint32_t _0;
  };

  struct U64_Body {
    uint64_t _0;
  };

  struct USize_Body {
    uintptr_t _0;
  };

  struct F32_Body {
    float _0;
  };

  struct F64_Body {
    double _0;
  };

  struct Char_Body {
    uint32_t _0;
  };

  struct Bool_Body {
    bool _0;
  };

  struct String_Body {
    // String _0;
    //  HACK: these values have the same layout as a String
    //  but don't have copy/destruct semantics. This allows
    //  `Value` to be a value type with simple semantics. When
    //  we actually implement Strings they will use internment
    uintptr_t _0;
    uint8_t *_1;
  };

  struct Symbol_Body {
    uintptr_t _0;
  };

  struct Entity_Body {
    uint64_t _0;
  };

  Tag tag;
  union {
    I8_Body i8;
    I16_Body i16;
    I32_Body i32;
    I64_Body i64;
    ISize_Body i_size;
    U8_Body u8;
    U16_Body u16;
    U32_Body u32;
    U64_Body u64;
    USize_Body u_size;
    F32_Body f32;
    F64_Body f64;
    Char_Body char_;
    Bool_Body bool_;
    String_Body string;
    Symbol_Body symbol;
    Entity_Body entity;
  };

 public:
  __host__ __device__ Value();

  __host__ __device__ Value(Tag tag, int8_t value);
  __host__ __device__ Value(Tag tag, int16_t value);
  __host__ __device__ Value(Tag tag, int32_t value);
  __host__ __device__ Value(Tag tag, int64_t value);
  __host__ __device__ Value(Tag tag, uint8_t value);
  __host__ __device__ Value(Tag tag, uint16_t value);
  __host__ __device__ Value(Tag tag, uint32_t value);
  __host__ __device__ Value(Tag tag, uint64_t value);
  __host__ __device__ Value(Tag tag, float value);
  __host__ __device__ Value(Tag tag, double value);
  __host__ __device__ Value(Tag tag, bool value);

  __host__ __device__ Value(Tag tag, const void *value);

  __host__ __device__ static Value None();
  __host__ __device__ static Value I8(int8_t value);
  __host__ __device__ static Value I16(int16_t value);
  __host__ __device__ static Value I32(int32_t value);
  __host__ __device__ static Value I64(int64_t value);
  __host__ __device__ static Value ISize(intptr_t value);
  __host__ __device__ static Value U8(uint8_t value);
  __host__ __device__ static Value U16(uint16_t value);
  __host__ __device__ static Value U32(uint32_t value);
  __host__ __device__ static Value U64(uint64_t value);
  __host__ __device__ static Value USize(uintptr_t value);
  __host__ __device__ static Value F32(float value);
  __host__ __device__ static Value F64(double value);
  __host__ __device__ static Value Char(uint32_t value);
  __host__ __device__ static Value Bool(bool value);

  template <typename T>
  __host__ __device__ T downcast() const {
    if constexpr (std::is_same_v<T, uint8_t>) {
      if (SAFETY) {
        if (this->type().size() != 1) {
          PANIC("downcast to char failed");
        }
      }
      return this->char_._0;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      if (SAFETY) {
        if (this->type().size() != 2) {
          PANIC("downcast to short failed");
        }
      }
      return this->i16._0;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      if (SAFETY) {
        if (this->type().size() != 4) {
          PANIC("downcast to int failed");
        }
      }
      return this->i32._0;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      if (SAFETY) {
        if (this->type().size() != 8) {
          PANIC("downcast to long long failed");
        }
      }
      return this->i64._0;
    } else if constexpr (std::is_same_v<T, uintptr_t>) {
      if (SAFETY) {
        if (this->type().size() != sizeof(uintptr_t)) {
          PANIC("downcast to uintptr_t failed");
        }
      }
      return this->i_size._0;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      if (SAFETY) {
        if (this->type().size() != 1) {
          PANIC("downcast to char failed");
        }
      }
      return this->char_._0;
    } else if constexpr (std::is_same_v<T, int16_t>) {
      if (SAFETY) {
        if (this->type().size() != 2) {
          PANIC("downcast to short failed");
        }
      }
      return this->i16._0;
    } else if constexpr (std::is_same_v<T, int32_t>) {
      if (SAFETY) {
        if (this->type().size() != 4) {
          PANIC("downcast to int failed");
        }
      }
      return this->i32._0;
    } else if constexpr (std::is_same_v<T, int64_t>) {
      if (SAFETY) {
        if (this->type().size() != 8) {
          PANIC("downcast to long long failed");
        }
      }
      return this->i64._0;
    } else if constexpr (std::is_same_v<T, intptr_t>) {
      if (SAFETY) {
        if (this->type().size() != sizeof(uintptr_t)) {
          PANIC("downcast to uintptr_t failed");
        }
      }
      return this->i_size._0;
    } else if constexpr (std::is_same_v<T, float>) {
      if (SAFETY) {
        if (this->type().size() != 4) {
          PANIC("downcast to float failed");
        }
      }
      return this->f32._0;
    } else if constexpr (std::is_same_v<T, double>) {
      if (SAFETY) {
        if (this->type().size() != 8) {
          PANIC("downcast to double failed");
        }
      }
      return this->f64._0;
    } else {
      PANIC("downcast to unknown type");
    }
  }

  __host__ __device__ Value(const Value &value);

  __host__ __device__ Value operator=(const Value &other);

  __host__ __device__ ValueType type() const;

  uint32_t to_u32();

  __host__ __device__ bool operator<(const Value &other) const;

  __host__ __device__ bool operator>(const Value &other) const;

  __host__ __device__ bool operator==(const Value &other) const;

  __host__ __device__ bool operator!=(const Value &other) const;

  __host__ __device__ Value operator+(const Value &other) const;

  __host__ __device__ Value operator-(const Value &other) const;

  __host__ __device__ size_t hash() const;

  friend std::ostream &operator<<(std::ostream &os, const Value &value);
};

template <typename Prov>
class Relation {
 private:
  String predicate_;
  ForeignTupleType schema_;
  uintptr_t size_;
  Array<typename Prov::FFITag> tags_;
  Array<Array<Value>> tuples_;

 public:
  Relation(String &predicate, ForeignTupleType schema, uintptr_t size,
           Array<typename Prov::FFITag> &&tags, Array<Array<Value>> &&tuples)
      : predicate_(predicate),
        schema_(schema),
        size_(size),
        tags_(std::move(tags)),
        tuples_(std::move(tuples)) {}

  Relation(const Relation<Prov> &relation)
      : predicate_(relation.predicate_),
        schema_(relation.schema_),
        size_(relation.size_),
        tags_(relation.tags_),
        tuples_(relation.tuples_) {}

  const String &predicate() const { return this->predicate_; }
  const ForeignTupleType &schema() const { return this->schema_; }
  uintptr_t size() const { return this->size_; }
  const Array<typename Prov::FFITag> &tags() const { return this->tags_; }
  const Array<Array<Value>> &tuples() const { return this->tuples_; }

  friend std::ostream &operator<<(std::ostream &os,
                                  const Relation<Prov> &relation) {
    return os << "Relation{" << std::endl
              << "predicate: " << relation.predicate() << std::endl
              << "schema: " << relation.schema() << std::endl
              << "size: " << relation.size() << std::endl
              << "tags: " << relation.tags() << std::endl
              << "tuples: " << relation.tuples() << std::endl
              << "}";
  }
};

template <typename Prov>
class StaticDB {
 private:
  Array<Relation<Prov>> relations_;

 public:
  StaticDB(Array<Relation<Prov>> &&relations)
      : relations_(std::move(relations)) {}

  const Array<Relation<Prov>> &relations() const { return this->relations_; }

  friend std::ostream &operator<<(std::ostream &os, const StaticDB &idb) {
    return os << "StaticDB(" << idb.relations() << ")";
  }
};

class TupleAccessor {
 private:
  int8_t len;
  int8_t indices[TUPLE_ACCESSOR_DEPTH];

 public:
  TupleAccessor(int8_t len, int8_t *indices);
  int8_t operator[](int8_t index);

  size_t to_index(TupleType schema) const;
  ValueType to_type(TupleType schema) const;

  TupleType result_type(TupleType schema) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const TupleAccessor &tuple_accessor);
};

enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  And,
  Or,
  Xor,
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
};

class Expr;

class BinaryExpr {
 public:
  BinaryOp op;
  const Expr *op1;
  const Expr *op2;

  BinaryExpr(BinaryOp op, const Expr &op1, const Expr &op2);
  BinaryExpr(const BinaryExpr &binary_expr);
  BinaryExpr &operator=(const BinaryExpr &other);
  ~BinaryExpr();

  TupleType result_type(TupleType schema) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const BinaryExpr &binary_expr);
};

class Expr {
 public:
  enum class Tag {
    Tuple,
    Access,
    Constant,
    Binary,
  };

  struct Tuple_Body {
    Array<Expr> _0;
  };

  struct Access_Body {
    TupleAccessor _0;
  };

  struct Constant_Body {
    Value _0;
  };
  struct Binary_Body {
    BinaryExpr _0;
  };

  Tag tag;
  union {
    Tuple_Body tuple;
    Access_Body access;
    Constant_Body constant;
    Binary_Body binary;
  };

  Expr(Array<Expr> tuple);
  Expr(TupleAccessor access);
  Expr(Value constant);
  Expr(BinaryExpr binary);

  Expr(const Expr &expr);

  Expr &operator=(const Expr &other);
  ~Expr();

  TupleType result_type(TupleType schema) const;

  bool is_permutation() const;

  bool is_constant() const;

  friend std::ostream &operator<<(std::ostream &os, const Expr &expr);
};

class Dataflow {
 public:
  enum class Tag {
    Unit,
    Relation,
    Project,
    Filter,
    Find,
    Union,
    Join,
    Intersect,
    Product,
    Antijoin,
    Difference,
    OverwriteOne
  };

  struct Unit_Body {
    ForeignTupleType _0;
  };

  struct Relation_Body {
    String _0;
  };

  struct Project_Body {
    Dataflow *_0;
    Expr _1;
  };

  struct Filter_Body {
    Dataflow *_0;
    Expr _1;
  };

  struct Find_Body {
    Dataflow *_0;
    Value _1;
  };

  struct Union_Body {
    Dataflow *_0;
    Dataflow *_1;
  };

  struct Join_Body {
    Dataflow *_0;
    Dataflow *_1;
    bool index_on_right;
  };

  struct Intersect_Body {
    Dataflow *_0;
    Dataflow *_1;
  };

  struct Product_Body {
    Dataflow *_0;
    Dataflow *_1;
  };

  struct Antijoin_Body {
    Dataflow *_0;
    Dataflow *_1;
  };

  struct Difference_Body {
    Dataflow *_0;
    Dataflow *_1;
  };

  struct OverwriteOne_Body {
    Dataflow *_0;
  };

  Tag tag;
  union {
    Unit_Body unit;
    Relation_Body relation;
    Project_Body project;
    Filter_Body filter;
    Find_Body find;
    Union_Body union_;
    Join_Body join;
    Intersect_Body intersect;
    Product_Body product;
    Antijoin_Body antijoin;
    Difference_Body difference;
    OverwriteOne_Body overwrite_one;
  };

  Dataflow(const Dataflow &dataflow);

  ~Dataflow();

  // Returns all relations that are used anywhere in this dataflow
  std::vector<std::string> dependencies() const;

  friend std::ostream &operator<<(std::ostream &os, const Dataflow &dataflow);
};

class Update {
 private:
  String target_;
  Dataflow dataflow_;

 public:
  Update(String target, Dataflow dataflow)
      : target_(target), dataflow_(dataflow) {}

  const String &target() const { return this->target_; }
  const Dataflow &dataflow() const { return this->dataflow_; }

  // Returns all relations that are used anywhere in this update
  std::vector<std::string> dependencies() const {
    return this->dataflow().dependencies();
  }

  friend std::ostream &operator<<(std::ostream &os, const Update &update) {
    return os << "Update{" << std::endl
              << "target: " << update.target() << std::endl
              << "dataflow: " << update.dataflow() << std::endl
              << "}";
  }
};

template <typename Prov>
struct Stratum {
 private:
  Array<String> relation_names_;
  Array<Relation<Prov>> relations_;
  Array<Update> updates_;

 public:
  Stratum(Array<String> relation_names, Array<Relation<Prov>> relations,
          Array<Update> updates)
      : relation_names_(relation_names),
        relations_(relations),
        updates_(updates) {}

  const Array<String> &relation_names() const { return this->relation_names_; }
  const Array<Relation<Prov>> &relations() const { return this->relations_; }
  const Array<Update> &updates() const { return this->updates_; }

  std::vector<std::string> dependencies() const {
    std::vector<std::string> deps;
    for (size_t i = 0; i < this->updates().size(); i++) {
      std::vector<std::string> update_deps = this->updates()[i].dependencies();
      deps.insert(deps.end(), update_deps.begin(), update_deps.end());
    }
    return deps;
  }
  bool is_recursive() const {
    auto dependencies = this->dependencies();
    auto relations = this->relation_names();
    bool is_recursive = false;
    for (size_t i = 0; i < relations.size(); i++) {
      if (std::find(dependencies.begin(), dependencies.end(), relations[i].to_string()) !=
          dependencies.end()) {
        is_recursive = true;
        break;
      }
    }
    return is_recursive;
  }

  friend std::ostream &operator<<(std::ostream &os, const Stratum &stratum) {
    return os << "Stratum{" << std::endl
              << "relation_names: " << stratum.relation_names() << std::endl
              << "relations: " << stratum.relations() << std::endl
              << "updates: " << stratum.updates() << std::endl
              << "}";
  }
};

struct Provenance {
  enum class Tag {
    Unit,
    MinMaxProb,
    DiffMinMaxProb,
    DiffAddMultProb,
    DiffTopKProofs
  };
  struct Unit {
  };
  struct MinMaxProb {
    double valid_threshold;
  };
  struct DiffMinMaxProb {
    // Array<void *, ForeignAlloc> storage;
    double valid_threshold;
  };
  struct DiffAddMultProb {
    // Array<void *, ForeignAlloc> storage;
    double valid_threshold;
  };
  struct DiffTopKProofs {
    Array<Array<float>> literal_probabilities;
  };

  Tag tag;
  union {
    Unit unit;
    MinMaxProb min_max_prob;
    DiffMinMaxProb diff_min_max_prob;
    DiffAddMultProb diff_add_mult_prob;
    DiffTopKProofs diff_top_k_proofs;
  };
};

struct C_Gradient {
  Array<uintptr_t> indices;
  Array<double> values;
  friend std::ostream &operator<<(std::ostream &os,
                                  const C_Gradient &gradient) {
    return os << "C_Gradient{" << gradient.indices << ", " << gradient.values
              << "}";
  }
};

struct C_DualNumber {
  double real;
  C_Gradient gradient;
  friend std::ostream &operator<<(std::ostream &os,
                                  const C_DualNumber &dual_number) {
    return os << "C_DualNumber{" << dual_number.real << ", "
              << dual_number.gradient << "}";
  }
};

struct C_Proof {
  Array<int32_t> literals;
  bool empty;
  friend std::ostream &operator<<(std::ostream &os, const C_Proof &proof) {
    return os << "C_Proof{" << proof.literals << "}";
  }
};
