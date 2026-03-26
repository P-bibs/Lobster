#include <thrust/copy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "alloc.h"
#include "bindings.h"
#include "utils.h"

__host__ __device__ ValueType::ValueType() : tag_(Tag::None) {}

__host__ __device__ bool ValueType::operator==(const ValueType &other) const {
  return this->tag_ == other.tag_;
}
__host__ __device__ bool ValueType::operator!=(const ValueType &other) const {
  return this->tag_ != other.tag_;
}

__host__ __device__ size_t ValueType::size() const {
  switch (this->tag_) {
    case Tag::None:
      return 0;
    case Tag::I8:
      return sizeof(int8_t);
    case Tag::I16:
      return sizeof(int16_t);
    case Tag::I32:
      return sizeof(int32_t);
    case Tag::I64:
      return sizeof(int64_t);
    case Tag::ISize:
      return sizeof(intptr_t);
    case Tag::U8:
      return sizeof(uint8_t);
    case Tag::U16:
      return sizeof(uint16_t);
    case Tag::U32:
      return sizeof(uint32_t);
    case Tag::U64:
      return sizeof(uint64_t);
    case Tag::USize:
      return sizeof(uintptr_t);
    case Tag::F32:
      return sizeof(float);
    case Tag::F64:
      return sizeof(double);
    case Tag::Char:
      return sizeof(uint32_t);
    case Tag::Bool:
      return sizeof(bool);
    case Tag::Str:
      return sizeof(uintptr_t);
    // case Tag::String:
    //   return sizeof(uintptr_t);
    case Tag::Symbol:
      return sizeof(uintptr_t);
    case Tag::Entity:
      return sizeof(uintptr_t);
    default:
      PANIC("unreachable");
  }
}

std::ostream &operator<<(std::ostream &os, const ValueType &value_type) {
  switch (value_type.tag()) {
    case ValueType::Tag::None:
      return os << "None";
    case ValueType::Tag::I8:
      return os << "I8";
    case ValueType::Tag::I16:
      return os << "I16";
    case ValueType::Tag::I32:
      return os << "I32";
    case ValueType::Tag::I64:
      return os << "I64";
    case ValueType::Tag::ISize:
      return os << "ISize";
    case ValueType::Tag::U8:
      return os << "U8";
    case ValueType::Tag::U16:
      return os << "U16";
    case ValueType::Tag::U32:
      return os << "U32";
    case ValueType::Tag::U64:
      return os << "U64";
    case ValueType::Tag::USize:
      return os << "USize";
    case ValueType::Tag::F32:
      return os << "F32";
    case ValueType::Tag::F64:
      return os << "F64";
    case ValueType::Tag::Char:
      return os << "Char";
    case ValueType::Tag::Bool:
      return os << "Bool";
    case ValueType::Tag::Str:
      return os << "Str";
    case ValueType::Tag::String:
      return os << "String";
    case ValueType::Tag::Symbol:
      return os << "Symbol";
    case ValueType::Tag::Entity:
      return os << "Entity";
    default:
      throw std::runtime_error("unreachable");
  }
}

String::String(uintptr_t length, uint8_t *value)
    : length_(length), value_(value) {}
String::String(const String &string)
    : length_(string.length_),
      value_(HostAlloc::template malloc<uint8_t>(string.length_)) {
  std::copy(string.value_, string.value_ + string.length_, this->value_);
}
String &String::operator=(const String &other) {
  this->length_ = other.length_;
  this->value_ = HostAlloc::template malloc<uint8_t>(other.length_);
  std::copy(other.value_, other.value_ + other.length_, this->value_);
  return *this;
}
String::~String() { HostAlloc::template destroy<uint8_t>(this->value_); }

std::string String::to_string() const {
  return std::string(reinterpret_cast<char *>(this->value_), this->length_);
}

uintptr_t String::length() const { return this->length_; }
const uint8_t *String::data() const { return this->value_; }

std::ostream &operator<<(std::ostream &os, const String &string) {
  return os << string.to_string();
}

ForeignTupleType::ForeignTupleType(Array<ForeignTupleType> tuple)
    : tag(Tag::Tuple), tuple({tuple}) {}
ForeignTupleType::ForeignTupleType(std::vector<ForeignTupleType> tuple)
    : tag(Tag::Tuple), tuple({Array<ForeignTupleType>(tuple)}) {}
ForeignTupleType::ForeignTupleType(ValueType unit)
    : tag(Tag::Unit), unit({unit}) {}

ForeignTupleType::ForeignTupleType(const ForeignTupleType &tuple_type)
    : tag(tuple_type.tag) {
  if (tuple_type.tag == Tag::Tuple) {
    new (&(this->tuple._0)) Array<ForeignTupleType>(tuple_type.tuple._0);
  } else if (tuple_type.tag == Tag::Unit) {
    this->unit._0 = tuple_type.unit._0;
  } else {
    throw std::runtime_error("unreachable");
  }
}
ForeignTupleType &ForeignTupleType::operator=(const ForeignTupleType &other) {
  if (this->tag == Tag::Tuple) {
    this->tuple._0.~Array();
  }

  this->tag = other.tag;
  if (other.tag == Tag::Tuple) {
    this->tuple._0 = other.tuple._0.clone();
  } else if (other.tag == Tag::Unit) {
    this->unit._0 = other.unit._0;
  } else {
    throw std::runtime_error("unreachable");
  }
  return *this;
}
ForeignTupleType::~ForeignTupleType() {
  if (this->tag == Tag::Tuple) {
    this->tuple._0.~Array();
  }
}
ForeignTupleType ForeignTupleType::clone() const {
  return ForeignTupleType(*this);
}

bool ForeignTupleType::operator==(const ForeignTupleType &other) const {
  if (this->tag != other.tag) {
    return false;
  }
  if (this->tag == Tag::Tuple) {
    return this->tuple._0 == other.tuple._0;
  } else if (this->tag == Tag::Unit) {
    return this->unit._0 == other.unit._0;
  } else {
    throw std::runtime_error("unreachable");
  }
}
bool ForeignTupleType::operator!=(const ForeignTupleType &other) const {
  return !(*this == other);
}

size_t ForeignTupleType::width() const {
  switch (this->tag) {
    case Tag::Unit:
      return 1;
    case Tag::Tuple:
      size_t width = 0;
      for (size_t i = 0; i < this->tuple._0.size(); i++) {
        width += this->tuple._0[i].width();
      }
      return width;
  }
  throw std::runtime_error("unreachable");
}

std::pair<ForeignTupleType, Array<ForeignTupleType>>
ForeignTupleType::remove_first() const {
  switch (this->tag) {
    case Tag::Unit:
      throw std::runtime_error("cannot remove first from unit");
    case Tag::Tuple:
      if (this->tuple._0.size() == 0) {
        throw std::runtime_error("cannot remove first from empty tuple");
      }
      ForeignTupleType first_column = this->tuple._0[0];
      Array<ForeignTupleType> rest_columns(this->tuple._0.size() - 1,
                                           this->tuple._0.data() + 1);
      return std::make_pair(first_column, rest_columns);
  }
  throw std::runtime_error("unreachable");
}

ForeignTupleType ForeignTupleType::append(const ForeignTupleType &other) const {
  if (this->tag == Tag::Unit) {
    if (other.tag == Tag::Unit) {
      std::vector<ForeignTupleType> body{*this, other};
      Array<ForeignTupleType> new_tuple(body);
      return ForeignTupleType(new_tuple);
    } else if (other.tag == Tag::Tuple) {
      Array<ForeignTupleType> new_tuple(*this, other.tuple._0);
      return ForeignTupleType(new_tuple);
    } else {
      throw std::runtime_error("unreachable");
    }
  } else if (this->tag == Tag::Tuple) {
    if (other.tag == Tag::Unit) {
      return Array(this->tuple._0, other);
    } else if (other.tag == Tag::Tuple) {
      return Array(this->tuple._0, other.tuple._0);
    } else {
      throw std::runtime_error("unreachable");
    }
  } else {
    throw std::runtime_error("unreachable");
  }
}

std::ostream &operator<<(std::ostream &os, const ForeignTupleType &tuple_type) {
  switch (tuple_type.tag) {
    case ForeignTupleType::Tag::Tuple:
      return os << "Tuple(" << tuple_type.tuple._0 << ")";
    case ForeignTupleType::Tag::Unit:
      return os << "Unit(" << tuple_type.unit._0 << ")";
    default:
      throw std::runtime_error("unreachable");
  }
}

__host__ __device__ TupleType::TupleType() : width_(0), valid_bytes_(2) {
  data_[0] = OPEN_PAREN;
  data_[1] = CLOSE_PAREN;
}
__host__ __device__ TupleType::TupleType(const TupleType &other, size_t start) {
  if (SAFETY) {
    if (start >= other.valid_bytes()) {
      PANIC("start out of range");
    }
  }
  auto kind = get_kind(other.data_[start]);
  if (kind == Kind::OpenParen) {
    auto close_paren = other.find_close_paren(start);
    auto valid_bytes = close_paren - start + 1;
    auto width = 0;
    for (size_t i = start; i < close_paren; i++) {
      if (get_kind(other.data_[i]) == Kind::Type) {
        width++;
      }
    }
    valid_bytes_ = valid_bytes;
    width_ = width;
    // TODO: somehow replace this with device-agnostic memcpy?
    for (auto ptr = other.data_ + start; ptr < other.data_ + close_paren + 1;
         ptr++) {
      data_[ptr - other.data_ - start] = *ptr;
    }
  } else if (kind == Kind::Type) {
    valid_bytes_ = 1;
    width_ = 1;
    data_[0] = other.data_[start];
  } else {
    PANIC("unreachable");
  }
}

__host__ __device__ TupleType::Kind TupleType::get_kind(uint8_t byte) {
  if (byte == OPEN_PAREN) {
    return TupleType::Kind::OpenParen;
  } else if (byte == CLOSE_PAREN) {
    return TupleType::Kind::CloseParen;
  } else {
    return TupleType::Kind::Type;
  }
}
__host__ __device__ uint8_t TupleType::find_close_paren(size_t start) const {
  if (SAFETY) {
    if (get_kind(data_[start]) != Kind::OpenParen) {
      PANIC("find_close_paren: start must be an open paren");
    }
  }
  size_t depth = 1;
  for (size_t i = start + 1; i < valid_bytes(); i++) {
    if (get_kind(data_[i]) == Kind::OpenParen) {
      depth++;
    } else if (get_kind(data_[i]) == Kind::CloseParen) {
      depth--;
      if (depth == 0) {
        return i;
      }
    }
  }
  PANIC("unmatched open paren");
}
__host__ __device__ uint8_t TupleType::valid_bytes() const {
  return valid_bytes_;
}
__host__ __device__ size_t TupleType::index_to_location(size_t index) const {
  size_t location = 1;
  while (index > 0) {
    if (location >= valid_bytes()) {
      PANIC("index out of range");
    }

    auto kind = get_kind(data_[location]);
    if (kind == Kind::OpenParen) {
      location = find_close_paren(location);
      location += 1;
      index -= 1;
    } else if (kind == Kind::Type) {
      location += 1;
      index -= 1;
    } else if (kind == Kind::CloseParen) {
      PANIC("unmatched close paren");
    }
  }
  return location;
}

__host__ __device__ void TupleType::validate() {
  if (!SAFETY) {
    return;
  }
  if (valid_bytes() == 0) {
    PANIC("Size must be at least 1, but is 0");
  }
  if (valid_bytes() > TupleType::SIZE) {
    PANIC("Size must be at most 30, but is %d", valid_bytes());
  }
  if (width() > TupleType::SIZE - 2) {
    PANIC("Width must be at most 30, but is %zu", width());
  }
  size_t num_values = 0;
  int depth = 0;
  for (size_t i = 0; i < valid_bytes(); i++) {
    if (get_kind(data_[i]) == Kind::Type) {
      if (data_[i] == static_cast<uint8_t>(ValueType::Tag::None)) {
        PANIC("None is not a valid value");
      }
      num_values++;
    }
    if (get_kind(data_[i]) == Kind::OpenParen) {
      depth++;
    }
    if (get_kind(data_[i]) == Kind::CloseParen) {
      depth--;
    }
  }
  if (num_values != width()) {
    PANIC("Number of values does not match width, %zu != %zu", num_values,
          width());
  }
  if (depth != 0) {
    PANIC("Unmatched parens, depth = %d", depth);
  }
}
TupleType::TupleType(ForeignTupleType tuple) : width_(0), valid_bytes_(0) {
  std::function<void(ForeignTupleType t)> l = [&](auto t) {
    if (t.tag == ForeignTupleType::Tag::Unit) {
      this->data_[valid_bytes_++] = static_cast<uint8_t>(t.unit._0.tag());
      width_ += 1;
    } else if (t.tag == ForeignTupleType::Tag::Tuple) {
      this->data_[valid_bytes_++] = OPEN_PAREN;
      for (size_t i = 0; i < t.tuple._0.size(); i++) {
        l(t.tuple._0[i]);
      }
      this->data_[valid_bytes_++] = CLOSE_PAREN;
    } else {
      throw std::runtime_error("unreachable");
    }
  };
  l(tuple);

  if (width_ != tuple.width()) {
    std::string message = "ForeignTupleType->TupleType: width mismatch: ";
    message += std::to_string(width_);
    message += " != ";
    message += std::to_string(tuple.width());
    throw std::runtime_error(message);
  }
  validate();
}
__host__ __device__ TupleType::TupleType(ValueType value)
    : width_(1), valid_bytes_(1) {
  data_[0] = static_cast<uint8_t>(value.tag());
  validate();
}
TupleType::TupleType(Array<TupleType> tuple) : width_(0), valid_bytes_(0) {
  if (tuple.size() == 0) {
    width_ = 0;
    valid_bytes_ = 2;
    data_[0] = OPEN_PAREN;
    data_[1] = CLOSE_PAREN;
  } else {
    for (size_t i = 0; i < tuple.size(); i++) {
      width_ += tuple[i].width();
      valid_bytes_ += tuple[i].valid_bytes();
    }
    valid_bytes_ += 2;
    if (valid_bytes_ > TupleType::SIZE) {
      throw std::runtime_error("TupleType too large");
    }

    size_t write_to = 1;
    data_[0] = OPEN_PAREN;
    for (size_t i = 0; i < tuple.size(); i++) {
      for (size_t j = 0; j < tuple[i].valid_bytes(); j++) {
        data_[write_to] = tuple[i].data_[j];
        write_to++;
      }
    }
    data_[write_to] = CLOSE_PAREN;
  }
  validate();
}
TupleType::TupleType(std::vector<TupleType> tuple) : TupleType(Array(tuple)) {}

bool TupleType::operator==(const TupleType &other) const {
  if (this->width() != other.width()) {
    return false;
  }
  if (this->valid_bytes() != other.valid_bytes()) {
    return false;
  }
  for (size_t i = 0; i < valid_bytes(); i++) {
    if (this->data_[i] != other.data_[i]) {
      return false;
    }
  }
  return true;
}
bool TupleType::operator!=(const TupleType &other) const {
  return !(*this == other);
}

__host__ __device__ size_t TupleType::width() const { return width_; }

__host__ __device__ size_t TupleType::size() const {
  if (this->valid_bytes_ <= 1) {
    PANIC("size of scalar tuple");
  }

  int size = 0;
  int i = 1;
  while (i < valid_bytes_ - 1) {
    if (get_kind(data_[i]) == Kind::Type) {
      size += 1;
      i += 1;
    } else if (get_kind(data_[i]) == Kind::OpenParen) {
      i = find_close_paren(i) + 1;
      size += 1;
    } else {
      PANIC("unreachable");
    }
  }
  return size;
}

__host__ __device__ size_t TupleType::bytewidth() const {
  auto bytewidth = 0;
  for (size_t i = 0; i < valid_bytes_; i++) {
    if (get_kind(data_[i]) == Kind::Type) {
      uint32_t widened = data_[i];
      bytewidth += reinterpret_cast<const ValueType *>(&widened)->size();
    }
  }
  return bytewidth;
}

__host__ __device__ bool TupleType::is_nested() const {
  return width_ + 2 != valid_bytes_;
}
__host__ __device__ bool TupleType::is_scalar() const {
  return width_ == 1 && valid_bytes_ == 1;
}

__host__ __device__ ValueType TupleType::singleton() {
  if (this->width() != 1) {
    PANIC("TupleType must have width 1 to call singleton");
  }

  uint32_t widened = valid_bytes() == 1 ? data_[0] : data_[1];
  return *reinterpret_cast<const ValueType *>(&widened);
}

std::pair<TupleType, TupleType> TupleType::remove_first() const {
  if (this->width() == 0) {
    throw std::runtime_error("cannot remove first from empty tuple");
  }
  auto first_column = this->at(0);

  auto rest = TupleType(*this);
  if (get_kind(rest.data_[1]) == Kind::OpenParen) {
    std::copy(rest.data_ + first_column.valid_bytes() + 1,
              rest.data_ + rest.valid_bytes_, rest.data_ + 1);
    rest.valid_bytes_ -= first_column.valid_bytes();
    rest.width_ -= first_column.width();
  } else if (get_kind(rest.data_[1]) == Kind::Type) {
    std::copy(rest.data_ + 2, rest.data_ + rest.valid_bytes_, rest.data_ + 1);
    rest.valid_bytes_ -= 1;
    rest.width_ -= 1;
  } else {
    throw std::runtime_error("unreachable");
  }
  first_column.validate();
  rest.validate();
  return std::make_pair(first_column, rest);
}

TupleType TupleType::append(const TupleType &other) const {
  if (SAFETY) {
    if (this->width() + other.width() > 32) {
      throw std::runtime_error("TupleType too large");
    }
  }

  if (this->is_scalar()) {
    throw std::runtime_error("cannot append to scalar");
  }
  if (other.is_scalar()) {
    throw std::runtime_error("cannot append scalar");
  }

  TupleType result;

  result.data_[0] = OPEN_PAREN;
  for (auto i = 1; i < this->valid_bytes() - 1; i++) {
    result.data_[i] = this->data_[i];
  }
  for (auto i = 1; i < other.valid_bytes() - 1; i++) {
    result.data_[i + this->valid_bytes() - 2] = other.data_[i];
  }

  result.valid_bytes_ = this->valid_bytes() + other.valid_bytes() - 2;
  result.width_ = this->width() + other.width();
  result.data_[result.valid_bytes_ - 1] = CLOSE_PAREN;

  result.validate();
  return result;
}

TupleType TupleType::flatten() const {
  if (is_scalar()) {
    return *this;
  }

  auto write_to{1};
  TupleType result;

  for (size_t i = 0; i < valid_bytes(); i++) {
    if (get_kind(data_[i]) == Kind::Type) {
      result.data_[write_to] = data_[i];
      write_to++;
    }
  }
  result.valid_bytes_ = write_to + 1;
  result.width_ = width_;
  result.data_[0] = OPEN_PAREN;
  result.data_[write_to] = CLOSE_PAREN;

  result.validate();

  return result;
}
std::vector<ValueType> TupleType::to_flat_vector() const {
  std::vector<ValueType> result;
  for (size_t i = 0; i < valid_bytes(); i++) {
    if (get_kind(data_[i]) == Kind::Type) {
      result.push_back(static_cast<ValueType::Tag>(data_[i]));
    }
  }
  return result;
}

ForeignTupleType TupleType::to_foreign() const {
  if (valid_bytes_ == 1) {
    return ForeignTupleType(static_cast<ValueType::Tag>(data_[0]));
  } else {
    std::vector<ForeignTupleType> body;
    auto size = this->size();
    for (size_t i = 0; i < size; i++) {
      body.push_back(this->at(i).to_foreign());
    }
    return ForeignTupleType(body);
  }
}

__host__ __device__ TupleType TupleType::at(size_t index) const {
  //if (SAFETY) {
  //  if (index >= width()) {
  //    PANIC("index out of range");
  //  }
  //}
  auto loc = index_to_location(index);
  TupleType result(*this, loc);
  result.validate();
  return result;
}

// Given a tuple accessor, returns the index of the column
// that is accessed (for use on flattened tables)
size_t TupleType::accessor_to_index(std::vector<int8_t> indices) const {
  if (indices.size() == 0) {
    return 0;
  }

  auto popped = indices;
  // pop first element
  popped.erase(popped.begin());

  auto index_in_child = this->at(indices[0]).accessor_to_index(popped);

  auto columns_before = 0;
  for (size_t i = 0; i < index_to_location(indices[0]); i++) {
    if (this->get_kind(data_[i]) == Kind::Type) {
      columns_before++;
    }
  }

  return columns_before + index_in_child;
}
ValueType TupleType::accessor_to_type(std::vector<int8_t> indices) const {
  auto index = accessor_to_index(indices);
  return this->flatten().at(index).singleton();
}

std::ostream &operator<<(std::ostream &os, const TupleType &tuple_type) {
  os << "TupleType{";
  for (size_t i = 0; i < tuple_type.valid_bytes(); i++) {
    auto kind = tuple_type.get_kind(tuple_type.data_[i]);
    if (kind == TupleType::Kind::OpenParen) {
      os << "(";
    } else if (kind == TupleType::Kind::CloseParen) {
      os << ")";
      if (i + 1 < tuple_type.valid_bytes()) {
        os << ",";
      }
    } else if (kind == TupleType::Kind::Type) {
      os << static_cast<ValueType::Tag>(tuple_type.data_[i]);
      os << ",";
    }
  }
  return os << "}";
}

Value::Tag Value::tag_from_type(ValueType::Tag t) {
  switch (t) {
    case ValueType::Tag::None:
      return Tag::None;
    case ValueType::Tag::I8:
      return Tag::I8;
    case ValueType::Tag::I16:
      return Tag::I16;
    case ValueType::Tag::I32:
      return Tag::I32;
    case ValueType::Tag::I64:
      return Tag::I64;
    case ValueType::Tag::ISize:
      return Tag::ISize;
    case ValueType::Tag::U8:
      return Tag::U8;
    case ValueType::Tag::U16:
      return Tag::U16;
    case ValueType::Tag::U32:
      return Tag::U32;
    case ValueType::Tag::U64:
      return Tag::U64;
    case ValueType::Tag::USize:
      return Tag::USize;
    case ValueType::Tag::F32:
      return Tag::F32;
    case ValueType::Tag::F64:
      return Tag::F64;
    case ValueType::Tag::Char:
      return Tag::Char;
    case ValueType::Tag::Bool:
      return Tag::Bool;
    case ValueType::Tag::String:
      return Tag::String;
    case ValueType::Tag::Symbol:
      return Tag::Symbol;
    case ValueType::Tag::Entity:
      return Tag::Entity;
    default:
      PANIC("tag_from_type: invalid type");
  }
}
__host__ __device__ Value::Value() : tag(Tag::None) {}

__host__ __device__ Value::Value(Tag tag, int8_t value) {
  if (tag == Tag::I8) {
    this->tag = Tag::I8;
    this->i8._0 = value;
  } else {
    PANIC("Value int8_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, int16_t value) {
  if (tag == Tag::I16) {
    this->tag = Tag::I16;
    this->i16._0 = value;
  } else {
    PANIC("Value int16_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, int32_t value) {
  if (tag == Tag::I32) {
    this->tag = Tag::I32;
    this->i32._0 = value;
  } else {
    PANIC("Value int32_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, int64_t value) {
  if (tag == Tag::I64) {
    this->tag = Tag::I64;
    this->i64._0 = value;
  } else if (tag == Tag::ISize) {
    this->tag = Tag::ISize;
    this->i_size._0 = value;
  } else {
    PANIC("Value int64_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, uint8_t value) {
  if (tag == Tag::U8) {
    this->tag = Tag::U8;
    this->u8._0 = value;
  } else {
    PANIC("Value uint8_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, uint16_t value) {
  if (tag == Tag::U16) {
    this->tag = Tag::U16;
    this->u16._0 = value;
  } else {
    PANIC("Value uint16_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, uint32_t value) {
  if (tag == Tag::U32) {
    this->tag = Tag::U32;
    this->u32._0 = value;
  } else if (tag == Tag::Char) {
    this->tag = Tag::Char;
    this->char_._0 = value;
  } else {
    PANIC("Value uint32_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, uint64_t value) {
  if (tag == Tag::U64) {
    this->tag = Tag::U64;
    this->u64._0 = value;
  } else if (tag == Tag::USize) {
    this->tag = Tag::USize;
    this->u_size._0 = value;
  } else {
    PANIC("Value uint64_t ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, float value) {
  if (tag == Tag::F32) {
    this->tag = Tag::F32;
    this->f32._0 = value;
  } else {
    PANIC("Value float ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, double value) {
  if (tag == Tag::F64) {
    this->tag = Tag::F64;
    this->f64._0 = value;
  } else {
    PANIC("Value double ctor: invalid tag");
  }
}
__host__ __device__ Value::Value(Tag tag, bool value) {
  if (tag == Tag::Bool) {
    this->tag = Tag::Bool;
    this->bool_._0 = value;
  } else {
    PANIC("Value bool ctor: invalid tag");
  }
}

__host__ __device__ Value::Value(Tag tag, const void *value) {
  this->tag = tag;
  switch (tag) {
    case Tag::None:
      break;
    case Tag::I8:
      this->i8._0 = *reinterpret_cast<const int8_t *>(value);
      break;
    case Tag::I16:
      this->i16._0 = *reinterpret_cast<const int16_t *>(value);
      break;
    case Tag::I32:
      this->i32._0 = *reinterpret_cast<const int32_t *>(value);
      break;
    case Tag::I64:
      this->i64._0 = *reinterpret_cast<const int64_t *>(value);
      break;
    case Tag::ISize:
      this->i_size._0 = *reinterpret_cast<const intptr_t *>(value);
      break;
    case Tag::U8:
      this->u8._0 = *reinterpret_cast<const uint8_t *>(value);
      break;
    case Tag::U16:
      this->u16._0 = *reinterpret_cast<const uint16_t *>(value);
      break;
    case Tag::U32:
      this->u32._0 = *reinterpret_cast<const uint32_t *>(value);
      break;
    case Tag::U64:
      this->u64._0 = *reinterpret_cast<const uint64_t *>(value);
      break;
    case Tag::USize:
      this->u_size._0 = *reinterpret_cast<const uintptr_t *>(value);
      break;
    case Tag::F32:
      this->f32._0 = *reinterpret_cast<const float *>(value);
      break;
    case Tag::F64:
      this->f64._0 = *reinterpret_cast<const double *>(value);
      break;
    case Tag::Char:
      this->char_._0 = *reinterpret_cast<const uint32_t *>(value);
      break;
    case Tag::Bool:
      this->bool_._0 = *reinterpret_cast<const bool *>(value);
      break;
    case Tag::String:
      PANIC("Value void* ctor: string is unimplemented");
      break;
    case Tag::Symbol:
      this->symbol._0 = *reinterpret_cast<const uintptr_t *>(value);
      break;
    case Tag::Entity:
      this->entity._0 = *reinterpret_cast<const uint64_t *>(value);
      break;
    default:
      PANIC("Value void* ctor: invalid tag");
  }
}

__host__ __device__ Value Value::None() { return Value(); }
__host__ __device__ Value Value::I8(int8_t value) {
  return Value(Tag::I8, value);
}
__host__ __device__ Value Value::I16(int16_t value) {
  return Value(Tag::I16, value);
}
__host__ __device__ Value Value::I32(int32_t value) {
  return Value(Tag::I32, value);
}
__host__ __device__ Value Value::I64(int64_t value) {
  return Value(Tag::I64, value);
}
__host__ __device__ Value Value::ISize(intptr_t value) {
  return Value(Tag::ISize, value);
}
__host__ __device__ Value Value::U8(uint8_t value) {
  return Value(Tag::U8, value);
}
__host__ __device__ Value Value::U16(uint16_t value) {
  return Value(Tag::U16, value);
}
__host__ __device__ Value Value::U32(uint32_t value) {
  return Value(Tag::U32, value);
}
__host__ __device__ Value Value::U64(uint64_t value) {
  return Value(Tag::U64, value);
}
__host__ __device__ Value Value::USize(uintptr_t value) {
  return Value(Tag::USize, value);
}
__host__ __device__ Value Value::F32(float value) {
  return Value(Tag::F32, value);
}
__host__ __device__ Value Value::F64(double value) {
  return Value(Tag::F64, value);
}
__host__ __device__ Value Value::Char(uint32_t value) {
  return Value(Tag::Char, value);
}
__host__ __device__ Value Value::Bool(bool value) {
  return Value(Tag::Bool, value);
}

__host__ __device__ Value::Value(const Value &value) : tag(value.tag) {
  // TODO: is it faster to do this with a memcpy instead of a branch?
  switch (value.tag) {
    case Tag::None:
      break;
    case Tag::I8:
      this->i8._0 = value.i8._0;
      break;
    case Tag::I16:
      this->i16._0 = value.i16._0;
      break;
    case Tag::I32:
      this->i32._0 = value.i32._0;
      break;
    case Tag::I64:
      this->i64._0 = value.i64._0;
      break;
    case Tag::ISize:
      this->i_size._0 = value.i_size._0;
      break;
    case Tag::U8:
      this->u8._0 = value.u8._0;
      break;
    case Tag::U16:
      this->u16._0 = value.u16._0;
      break;
    case Tag::U32:
      this->u32._0 = value.u32._0;
      break;
    case Tag::U64:
      this->u64._0 = value.u64._0;
      break;
    case Tag::USize:
      this->u_size._0 = value.u_size._0;
      break;
    case Tag::F32:
      this->f32._0 = value.f32._0;
      break;
    case Tag::F64:
      this->f64._0 = value.f64._0;
      break;
    case Tag::Char:
      this->char_._0 = value.char_._0;
      break;
    case Tag::Bool:
      this->bool_._0 = value.bool_._0;
      break;
    // case Tag::String:
    //   this->string._0 = value.string._0;
    //   break;
    case Tag::Symbol:
      this->symbol._0 = value.symbol._0;
      break;
    case Tag::Entity:
      this->entity._0 = value.entity._0;
      break;
    default:
      PANIC("Value copy ctor: invalid tag");
  }
}

__host__ __device__ Value Value::operator=(const Value &other) {
  this->tag = other.tag;
  switch (other.tag) {
    case Tag::None:
      break;
    case Tag::I8:
      this->i8._0 = other.i8._0;
      break;
    case Tag::I16:
      this->i16._0 = other.i16._0;
      break;
    case Tag::I32:
      this->i32._0 = other.i32._0;
      break;
    case Tag::I64:
      this->i64._0 = other.i64._0;
      break;
    case Tag::ISize:
      this->i_size._0 = other.i_size._0;
      break;
    case Tag::U8:
      this->u8._0 = other.u8._0;
      break;
    case Tag::U16:
      this->u16._0 = other.u16._0;
      break;
    case Tag::U32:
      this->u32._0 = other.u32._0;
      break;
    case Tag::U64:
      this->u64._0 = other.u64._0;
      break;
    case Tag::USize:
      this->u_size._0 = other.u_size._0;
      break;
    case Tag::F32:
      this->f32._0 = other.f32._0;
      break;
    case Tag::F64:
      this->f64._0 = other.f64._0;
      break;
    case Tag::Char:
      this->char_._0 = other.char_._0;
      break;
    case Tag::Bool:
      this->bool_._0 = other.bool_._0;
      break;
    // case Tag::String:
    //   this->string._0 = other.string._0;
    //   break;
    case Tag::Symbol:
      this->symbol._0 = other.symbol._0;
      break;
    case Tag::Entity:
      this->entity._0 = other.entity._0;
      break;
    default:
      PANIC("Value copy assignment: invalid tag");
  }
  return *this;
}

__host__ __device__ ValueType Value::type() const {
  switch (this->tag) {
    case Tag::None:
      return ValueType::None();
    case Tag::I8:
      return ValueType::I8();
    case Tag::I16:
      return ValueType::I16();
    case Tag::I32:
      return ValueType::I32();
    case Tag::I64:
      return ValueType::I64();
    case Tag::ISize:
      return ValueType::ISize();
    case Tag::U8:
      return ValueType::U8();
    case Tag::U16:
      return ValueType::U16();
    case Tag::U32:
      return ValueType::U32();
    case Tag::U64:
      return ValueType::U64();
    case Tag::USize:
      return ValueType::USize();
    case Tag::F32:
      return ValueType::F32();
    case Tag::F64:
      return ValueType::F64();
    case Tag::Char:
      return ValueType::Char();
    case Tag::Bool:
      return ValueType::Bool();
    // case Tag::String:
    //   return ValueType::String();
    case Tag::Symbol:
      return ValueType::Symbol();
    case Tag::Entity:
      return ValueType::Entity();
    default:
      PANIC("Value::type: invalid tag");
  }
}

uint32_t Value::to_u32() {
  if (this->tag == Tag::U8) {
    return this->u8._0;
  } else if (this->tag == Tag::U16) {
    return this->u16._0;
  } else if (this->tag == Tag::U32) {
    return this->u32._0;
  } else if (this->tag == Tag::USize) {
    return this->u_size._0;
  } else {
    throw std::runtime_error("unreachable");
  }
}

__host__ __device__ bool Value::operator<(const Value &other) const {
  if (this->tag != other.tag) {
    return false;
  }
  switch (this->tag) {
    case Tag::None:
      return false;
    case Tag::I8:
      return this->i8._0 < other.i8._0;
    case Tag::I16:
      return this->i16._0 < other.i16._0;
    case Tag::I32:
      return this->i32._0 < other.i32._0;
    case Tag::I64:
      return this->i64._0 < other.i64._0;
    case Tag::ISize:
      return this->i_size._0 < other.i_size._0;
    case Tag::U8:
      return this->u8._0 < other.u8._0;
    case Tag::U16:
      return this->u16._0 < other.u16._0;
    case Tag::U32:
      return this->u32._0 < other.u32._0;
    case Tag::U64:
      return this->u64._0 < other.u64._0;
    case Tag::USize:
      return this->u_size._0 < other.u_size._0;
    case Tag::F32:
      return this->f32._0 < other.f32._0;
    case Tag::F64:
      return this->f64._0 < other.f64._0;
    case Tag::Char:
      return this->char_._0 < other.char_._0;
    case Tag::Bool:
      return this->bool_._0 < other.bool_._0;
    // case Tag::String:
    //   PANIC("cannot compare strings");
    //   // return this->string._0 < other.string._0;
    case Tag::Symbol:
      return this->symbol._0 < other.symbol._0;
    case Tag::Entity:
      return this->entity._0 < other.entity._0;
    default:
      PANIC("unreachable");
  }
}

__host__ __device__ bool Value::operator>(const Value &other) const {
  if (this->tag != other.tag) {
    return false;
  }
  switch (this->tag) {
    case Tag::None:
      return false;
    case Tag::I8:
      return this->i8._0 > other.i8._0;
    case Tag::I16:
      return this->i16._0 > other.i16._0;
    case Tag::I32:
      return this->i32._0 > other.i32._0;
    case Tag::I64:
      return this->i64._0 > other.i64._0;
    case Tag::ISize:
      return this->i_size._0 > other.i_size._0;
    case Tag::U8:
      return this->u8._0 > other.u8._0;
    case Tag::U16:
      return this->u16._0 > other.u16._0;
    case Tag::U32:
      return this->u32._0 > other.u32._0;
    case Tag::U64:
      return this->u64._0 > other.u64._0;
    case Tag::USize:
      return this->u_size._0 > other.u_size._0;
    case Tag::F32:
      return this->f32._0 > other.f32._0;
    case Tag::F64:
      return this->f64._0 > other.f64._0;
    case Tag::Char:
      return this->char_._0 > other.char_._0;
    case Tag::Bool:
      return this->bool_._0 > other.bool_._0;
    // case Tag::String:
    //   PANIC("cannot compare strings");
    //   // return this->string._0 > other.string._0;
    case Tag::Symbol:
      return this->symbol._0 > other.symbol._0;
    case Tag::Entity:
      return this->entity._0 > other.entity._0;
    default:
      PANIC("unreachable");
  }
}

__host__ __device__ bool Value::operator==(const Value &other) const {
  if (this->tag != other.tag) {
    return false;
  }
  switch (this->tag) {
    case Tag::None:
      return true;
    case Tag::I8:
      return this->i8._0 == other.i8._0;
    case Tag::I16:
      return this->i16._0 == other.i16._0;
    case Tag::I32:
      return this->i32._0 == other.i32._0;
    case Tag::I64:
      return this->i64._0 == other.i64._0;
    case Tag::ISize:
      return this->i_size._0 == other.i_size._0;
    case Tag::U8:
      return this->u8._0 == other.u8._0;
    case Tag::U16:
      return this->u16._0 == other.u16._0;
    case Tag::U32:
      return this->u32._0 == other.u32._0;
    case Tag::U64:
      return this->u64._0 == other.u64._0;
    case Tag::USize:
      return this->u_size._0 == other.u_size._0;
    case Tag::F32:
      return this->f32._0 == other.f32._0;
    case Tag::F64:
      return this->f64._0 == other.f64._0;
    case Tag::Char:
      return this->char_._0 == other.char_._0;
    case Tag::Bool:
      return this->bool_._0 == other.bool_._0;
    // case Tag::String:
    //   return this->string._0 == other.string._0;
    case Tag::Symbol:
      return this->symbol._0 == other.symbol._0;
    case Tag::Entity:
      return this->entity._0 == other.entity._0;
    default:
      PANIC("unreachable");
  }
}

__host__ __device__ bool Value::operator!=(const Value &other) const {
  if (this->tag != other.tag) {
    return false;
  }
  switch (this->tag) {
    case Tag::None:
      return true;
    case Tag::I8:
      return this->i8._0 != other.i8._0;
    case Tag::I16:
      return this->i16._0 != other.i16._0;
    case Tag::I32:
      return this->i32._0 != other.i32._0;
    case Tag::I64:
      return this->i64._0 != other.i64._0;
    case Tag::ISize:
      return this->i_size._0 != other.i_size._0;
    case Tag::U8:
      return this->u8._0 != other.u8._0;
    case Tag::U16:
      return this->u16._0 != other.u16._0;
    case Tag::U32:
      return this->u32._0 != other.u32._0;
    case Tag::U64:
      return this->u64._0 != other.u64._0;
    case Tag::USize:
      return this->u_size._0 != other.u_size._0;
    case Tag::F32:
      return this->f32._0 != other.f32._0;
    case Tag::F64:
      return this->f64._0 != other.f64._0;
    case Tag::Char:
      return this->char_._0 != other.char_._0;
    case Tag::Bool:
      return this->bool_._0 != other.bool_._0;
    // case Tag::String:
    //   return this->string._0 != other.string._0;
    case Tag::Symbol:
      return this->symbol._0 != other.symbol._0;
    case Tag::Entity:
      return this->entity._0 != other.entity._0;
    default:
      PANIC("unreachable");
  }
}
__host__ __device__ Value Value::operator+(const Value &other) const {
  if (this->tag != other.tag) {
    PANIC("cannot subtract different types");
  }
  switch (this->tag) {
    case Tag::None:
      PANIC("cannot subtract None");
    case Tag::I8:
      return Value::I8(this->i8._0 + other.i8._0);
    case Tag::I16:
      return Value::I16(this->i16._0 + other.i16._0);
    case Tag::I32:
      return Value::I32(this->i32._0 + other.i32._0);
    case Tag::I64:
      return Value::I64(this->i64._0 + other.i64._0);
    case Tag::ISize:
      return Value::ISize(this->i_size._0 + other.i_size._0);
    case Tag::U8:
      return Value::U8(this->u8._0 + other.u8._0);
    case Tag::U16:
      return Value::U16(this->u16._0 + other.u16._0);
    case Tag::U32:
      return Value::U32(this->u32._0 + other.u32._0);
    case Tag::U64:
      return Value::U64(this->u64._0 + other.u64._0);
    case Tag::USize:
      // TODO: handle the overflow case better
      return Value::USize(this->u_size._0 + other.u_size._0);
    case Tag::F32:
      return Value::F32(this->f32._0 + other.f32._0);
    case Tag::F64:
      return Value::F64(this->f64._0 + other.f64._0);
    default:
      PANIC("Attempted to subtract invalid type");
  }
}
__host__ __device__ Value Value::operator-(const Value &other) const {
  if (this->tag != other.tag) {
    PANIC("cannot subtract different types");
  }
  switch (this->tag) {
    case Tag::None:
      PANIC("cannot subtract None");
    case Tag::I8:
      return Value::I8(this->i8._0 - other.i8._0);
    case Tag::I16:
      return Value::I16(this->i16._0 - other.i16._0);
    case Tag::I32:
      return Value::I32(this->i32._0 - other.i32._0);
    case Tag::I64:
      return Value::I64(this->i64._0 - other.i64._0);
    case Tag::ISize:
      return Value::ISize(this->i_size._0 - other.i_size._0);
    case Tag::U8:
      return Value::U8(this->u8._0 - other.u8._0);
    case Tag::U16:
      return Value::U16(this->u16._0 - other.u16._0);
    case Tag::U32:
      return Value::U32(this->u32._0 - other.u32._0);
    case Tag::U64:
      return Value::U64(this->u64._0 - other.u64._0);
    case Tag::USize: {
      // TODO: handle the overflow case better
      if (this->u_size._0 < other.u_size._0) {
        return Value::USize(0);
      }
      return Value::USize(this->u_size._0 - other.u_size._0);
    }
    case Tag::F32:
      return Value::F32(this->f32._0 - other.f32._0);
    case Tag::F64:
      return Value::F64(this->f64._0 - other.f64._0);
    default:
      PANIC("Attempted to subtract invalid type");
  }
}

__host__ __device__ size_t Value::hash() const {
  switch (this->tag) {
    case Tag::None:
      return 0;
    case Tag::I8:
      return simple_hash(this->i8._0);
    case Tag::I16:
      return simple_hash(this->i16._0);
    case Tag::I32:
      return simple_hash(this->i32._0);
    case Tag::I64:
      return simple_hash(this->i64._0);
    case Tag::ISize:
      return simple_hash(this->i_size._0);
    case Tag::U8:
      return simple_hash(this->u8._0);
    case Tag::U16:
      return simple_hash(this->u16._0);
    case Tag::U32:
      return simple_hash(this->u32._0);
    case Tag::U64:
      return simple_hash(this->u64._0);
    case Tag::USize:
      return simple_hash(this->u_size._0);
    case Tag::F32:
      return simple_hash(*reinterpret_cast<const uint32_t *>(&this->f32._0));
    case Tag::F64:
      return simple_hash(*reinterpret_cast<const uint64_t *>(&this->f64._0));
    case Tag::Char:
      return simple_hash(this->char_._0);
    case Tag::Bool:
      return simple_hash(this->bool_._0);
    // case Tag::String:
    //   return simple_hash(this->string._0);
    case Tag::Symbol:
      return simple_hash(this->symbol._0);
    case Tag::Entity:
      return simple_hash(this->entity._0);
    default:
      PANIC("unreachable");
  }
}

std::ostream &print_f32(std::ostream &os, float f) {
      os << "F32(" << std::to_string(f);
      //os << "|";
      //union {
      //  float f;
      //  uint32_t i;
      //} u;
      //u.f = f;
      //std::string s;
      //for (int i = 0; i < 32; i++) {
      //  if (u.i % 2)
      //    s.push_back('1');
      //  else
      //    s.push_back('0');
      //  u.i >>= 1;
      //}
      //std::string reversed(s.rbegin(), s.rend());
      //os << reversed;
      return os << ")";
}

std::ostream &operator<<(std::ostream &os, const Value &value) {
  switch (value.tag) {
    case Value::Tag::None:
      return os << "None";
    case Value::Tag::I8:
      return os << "I8(" << std::to_string(value.i8._0) << ")";
    case Value::Tag::I16:
      return os << "I16(" << std::to_string(value.i16._0) << ")";
    case Value::Tag::I32:
      return os << "I32(" << std::to_string(value.i32._0) << ")";
    case Value::Tag::I64:
      return os << "I64(" << std::to_string(value.i64._0) << ")";
    case Value::Tag::ISize:
      return os << "ISize(" << std::to_string(value.i_size._0) << ")";
    case Value::Tag::U8:
      return os << "U8(" << std::to_string(value.u8._0) << ")";
    case Value::Tag::U16:
      return os << "U16(" << std::to_string(value.u16._0) << ")";
    case Value::Tag::U32:
      return os << "U32(" << std::to_string(value.u32._0) << ")";
    case Value::Tag::U64:
      return os << "U64(" << std::to_string(value.u64._0) << ")";
    case Value::Tag::USize:
      return os << "USize(" << std::to_string(value.u_size._0) << ")";
    case Value::Tag::F32:
      return print_f32(os, value.f32._0);
    case Value::Tag::F64:
      return os << "F64(" << std::to_string(value.f64._0) << ")";
    case Value::Tag::Char:
      return os << "Char(" << std::to_string(value.char_._0) << ")";
    case Value::Tag::Bool:
      return os << "Bool(" << std::to_string(value.bool_._0) << ")";
    // case Value::Tag::String:
    //   return os << "String(" << value.string._0 << ")";
    case Value::Tag::Symbol:
      return os << "Symbol(" << std::to_string(value.symbol._0) << ")";
    case Value::Tag::Entity:
      return os << "Entity(" << std::to_string(value.entity._0) << ")";
    default:
      throw std::runtime_error("unreachable");
  }
}

TupleAccessor::TupleAccessor(int8_t len, int8_t *indices) : len(len) {
  for (int8_t i = 0; i < len; i++) {
    this->indices[i] = indices[i];
  }
}

int8_t TupleAccessor::operator[](int8_t index) {
  if (index >= this->len) {
    throw std::out_of_range("index out of range");
  }
  return this->indices[index];
}

size_t TupleAccessor::to_index(TupleType schema) const {
  return schema.accessor_to_index(
      std::vector<int8_t>(this->indices, this->indices + this->len));
}
ValueType TupleAccessor::to_type(TupleType schema) const {
  return schema.accessor_to_type(
      std::vector<int8_t>(this->indices, this->indices + this->len));
}

TupleType TupleAccessor::result_type(TupleType schema) const {
  if (this->len == 0) {
    throw std::runtime_error("invalid tuple accessor");
  }

  auto result = schema;
  for (int8_t i = 0; i < this->len; i++) {
    result = result.at(this->indices[i]);
  }
  return result;
}

std::ostream &operator<<(std::ostream &os,
                         const TupleAccessor &tuple_accessor) {
  if (tuple_accessor.len == 0) {
    throw std::runtime_error("invalid tuple accessor");
  }
  os << "TupleAccessor[";
  for (int i = 0; i < tuple_accessor.len; i++) {
    os << std::to_string(tuple_accessor.indices[i]);
    if (i < tuple_accessor.len - 1) {
      os << ", ";
    }
  }
  return os << "]";
}

BinaryExpr::BinaryExpr(BinaryOp op, const Expr &op1, const Expr &op2)
    : op(op), op1(new Expr(op1)), op2(new Expr(op2)) {}
BinaryExpr::BinaryExpr(const BinaryExpr &other) : op(other.op) {
  this->op1 = new Expr(*other.op1);
  this->op2 = new Expr(*other.op2);
}
BinaryExpr &BinaryExpr::operator=(const BinaryExpr &other) {
  this->op = other.op;
  delete this->op1;
  delete this->op2;
  this->op1 = new Expr(*other.op1);
  this->op2 = new Expr(*other.op2);
  return *this;
}

BinaryExpr::~BinaryExpr() {
  delete this->op1;
  delete this->op2;
}

TupleType BinaryExpr::result_type(TupleType schema) const {
  auto type1 = this->op1->result_type(schema);
  auto type2 = this->op2->result_type(schema);

  if (this->op == BinaryOp::Add || this->op == BinaryOp::Sub ||
      this->op == BinaryOp::Mul || this->op == BinaryOp::Div ||
      this->op == BinaryOp::Mod) {
    if (type1 != type2) {
      throw std::runtime_error("invalid binary operation");
    }
    return type1;
  } else if (this->op == BinaryOp::And || this->op == BinaryOp::Or ||
             this->op == BinaryOp::Eq || this->op == BinaryOp::Neq ||
             this->op == BinaryOp::Lt || this->op == BinaryOp::Leq ||
             this->op == BinaryOp::Gt || this->op == BinaryOp::Geq) {
    if (type1 != type2) {
      throw std::runtime_error("invalid binary operation");
    }
    return ValueType::Bool();
  } else {
    throw std::runtime_error("unreachable");
  }
}

std::ostream &operator<<(std::ostream &os, const BinaryExpr &binary_expr) {
  std::string op;
  switch (binary_expr.op) {
    case BinaryOp::Add:
      op = "+";
      break;
    case BinaryOp::Sub:
      op = "-";
      break;
    case BinaryOp::Mul:
      op = "*";
      break;
    case BinaryOp::Div:
      op = "/";
      break;
    case BinaryOp::Mod:
      op = "%";
      break;
    case BinaryOp::And:
      op = "&&";
      break;
    case BinaryOp::Or:
      op = "||";
      break;
    case BinaryOp::Eq:
      op = "==";
      break;
    case BinaryOp::Neq:
      op = "!=";
      break;
    case BinaryOp::Lt:
      op = "<";
      break;
    case BinaryOp::Leq:
      op = "<=";
      break;
    case BinaryOp::Gt:
      op = ">";
      break;
    case BinaryOp::Geq:
      op = ">=";
      break;
    default:
      throw std::runtime_error("unreachable");
  }
  return os << "BinaryExpr(" << op << ", " << *binary_expr.op1 << ", "
            << *binary_expr.op2 << ")";
}

Expr::Expr(Array<Expr> tuple) : tag(Tag::Tuple), tuple({tuple}) {}
Expr::Expr(TupleAccessor access) : tag(Tag::Access), access({access}) {}
Expr::Expr(Value constant) : tag(Tag::Constant), constant({constant}) {}
Expr::Expr(BinaryExpr binary) : tag(Tag::Binary), binary({binary}) {}

Expr::Expr(const Expr &expr) : tag(expr.tag) {
  if (expr.tag == Tag::Tuple) {
    new (&(this->tuple._0)) Array<Expr>(expr.tuple._0);
  } else if (expr.tag == Tag::Access) {
    this->access._0 = expr.access._0;
  } else if (expr.tag == Tag::Constant) {
    this->constant._0 = expr.constant._0;
  } else if (expr.tag == Tag::Binary) {
    new (&(this->binary._0)) BinaryExpr(expr.binary._0);
  } else {
    throw std::runtime_error("unreachable");
  }
}

Expr &Expr::operator=(const Expr &other) {
  if (this->tag == Tag::Tuple) {
    this->tuple._0.~Array();
  } else if (this->tag == Tag::Access) {
    this->access._0.~TupleAccessor();
  } else if (this->tag == Tag::Constant) {
    this->constant._0.~Value();
  } else if (this->tag == Tag::Binary) {
    this->binary._0.~BinaryExpr();
  }

  this->tag = other.tag;
  if (other.tag == Tag::Tuple) {
    new (&(this->tuple._0)) Array<Expr>(other.tuple._0);
  } else if (other.tag == Tag::Access) {
    this->access._0 = other.access._0;
  } else if (other.tag == Tag::Constant) {
    this->constant._0 = other.constant._0;
  } else if (other.tag == Tag::Binary) {
    new (&(this->binary._0)) BinaryExpr(other.binary._0);
  } else {
    throw std::runtime_error("unreachable");
  }
  return *this;
}
Expr::~Expr() {
  if (this->tag == Tag::Tuple) {
    this->tuple._0.~Array();
  } else if (this->tag == Tag::Access) {
    this->access._0.~TupleAccessor();
  } else if (this->tag == Tag::Constant) {
    this->constant._0.~Value();
  } else if (this->tag == Tag::Binary) {
    this->binary._0.~BinaryExpr();
  }
}

TupleType Expr::result_type(TupleType schema) const {
  if (this->tag == Tag::Tuple) {
    std::vector<TupleType> body;

    for (size_t i = 0; i < this->tuple._0.size(); i++) {
      body.push_back(this->tuple._0[i].result_type(schema));
    }
    return TupleType(body);
  } else if (this->tag == Tag::Access) {
    return this->access._0.result_type(schema);
  } else if (this->tag == Tag::Constant) {
    return TupleType(this->constant._0.type());
  } else if (this->tag == Tag::Binary) {
    return this->binary._0.result_type(schema);
  } else {
    throw std::runtime_error("unreachable");
  }
}

bool Expr::is_permutation() const {
  if (this->tag == Tag::Tuple) {
    for (size_t i = 0; i < this->tuple._0.size(); i++) {
      if (!this->tuple._0[i].is_permutation()) {
        return false;
      }
    }
    return true;
  } else if (this->tag == Tag::Access) {
    return true;
  } else if (this->tag == Tag::Constant) {
    return true;
  } else if (this->tag == Tag::Binary) {
    return false;
  } else {
    PANIC("unreachable");
  }
}

bool Expr::is_constant() const {
  if (this->tag == Tag::Tuple) {
    for (size_t i = 0; i < this->tuple._0.size(); i++) {
      if (!this->tuple._0[i].is_constant()) {
        return false;
      }
    }
    return true;
  } else if (this->tag == Tag::Access) {
    return false;
  } else if (this->tag == Tag::Constant) {
    return true;
  } else if (this->tag == Tag::Binary) {
    return false;
  } else {
    PANIC("unreachable");
  }
}

std::ostream &operator<<(std::ostream &os, const Expr &expr) {
  switch (expr.tag) {
    case Expr::Tag::Tuple:
      return os << "Tuple(" << expr.tuple._0 << ")";
    case Expr::Tag::Access:
      return os << "Access(" << expr.access._0 << ")";
    case Expr::Tag::Constant:
      return os << "Constant(" << expr.constant._0 << ")";
    case Expr::Tag::Binary:
      return os << "Binary(" << expr.binary._0 << ")";
    default:
      throw std::runtime_error("unreachable");
  }
}

Dataflow::Dataflow(const Dataflow &dataflow) : tag(dataflow.tag) {
  if (dataflow.tag == Tag::Unit) {
    this->unit._0 = dataflow.unit._0;
  } else if (dataflow.tag == Tag::Relation) {
    this->relation._0 = dataflow.relation._0;
  } else if (dataflow.tag == Tag::Project) {
    this->project._0 = dataflow.project._0;
    this->project._1 = dataflow.project._1;
  } else if (dataflow.tag == Tag::Filter) {
    this->filter._0 = dataflow.filter._0;
    this->filter._1 = dataflow.filter._1;
  } else if (dataflow.tag == Tag::Union) {
    this->union_._0 = dataflow.union_._0;
    this->union_._1 = dataflow.union_._1;
  } else if (dataflow.tag == Tag::Join) {
    this->join._0 = dataflow.join._0;
    this->join._1 = dataflow.join._1;
  } else if (dataflow.tag == Tag::Intersect) {
    this->intersect._0 = dataflow.intersect._0;
    this->intersect._1 = dataflow.intersect._1;
  } else if (dataflow.tag == Tag::Product) {
    this->product._0 = dataflow.product._0;
    this->product._1 = dataflow.product._1;
  } else if (dataflow.tag == Tag::Antijoin) {
    this->antijoin._0 = dataflow.antijoin._0;
    this->antijoin._1 = dataflow.antijoin._1;
  } else if (dataflow.tag == Tag::Difference) {
    this->difference._0 = dataflow.difference._0;
    this->difference._1 = dataflow.difference._1;
  } else if (dataflow.tag == Tag::OverwriteOne) {
    this->overwrite_one._0 = dataflow.overwrite_one._0;
  } else {
    throw std::runtime_error("unreachable");
  }
}

Dataflow::~Dataflow() {}

// Returns all relations that are used anywhere in this dataflow
std::vector<std::string> Dataflow::dependencies() const {
  std::vector<std::string> deps;
  std::vector<const Dataflow *> stack;
  stack.push_back(this);
  while (!stack.empty()) {
    const Dataflow *dataflow = stack.back();
    stack.pop_back();
    switch (dataflow->tag) {
      case Tag::Unit:
        break;
      case Tag::Relation:
        deps.push_back(dataflow->relation._0.to_string());
        break;
      case Tag::Project:
        stack.push_back(dataflow->project._0);
        break;
      case Tag::Filter:
        stack.push_back(dataflow->filter._0);
        break;
      case Tag::Union:
        stack.push_back(dataflow->union_._0);
        stack.push_back(dataflow->union_._1);
        break;
      case Tag::Join:
        stack.push_back(dataflow->join._0);
        stack.push_back(dataflow->join._1);
        break;
      case Tag::Intersect:
        stack.push_back(dataflow->intersect._0);
        stack.push_back(dataflow->intersect._1);
        break;
      case Tag::Product:
        stack.push_back(dataflow->product._0);
        stack.push_back(dataflow->product._1);
        break;
      case Tag::Antijoin:
        stack.push_back(dataflow->antijoin._0);
        stack.push_back(dataflow->antijoin._1);
        break;
      case Tag::Difference:
        stack.push_back(dataflow->difference._0);
        stack.push_back(dataflow->difference._1);
        break;
      case Tag::OverwriteOne:
        stack.push_back(dataflow->overwrite_one._0);
        break;
      case Tag::Find:
        stack.push_back(dataflow->find._0);
        break;
      default:
        throw std::runtime_error("unreachable");
    }
  }
  return deps;
}

std::ostream &operator<<(std::ostream &os, const Dataflow &dataflow) {
  switch (dataflow.tag) {
    case Dataflow::Tag::Unit:
      return os << "Unit(" << dataflow.unit._0 << ")";
    case Dataflow::Tag::Relation:
      return os << "Relation(" << dataflow.relation._0 << ")";
    case Dataflow::Tag::Project:
      return os << "Project(" << dataflow.project._0 << ", "
                << dataflow.project._1 << ")";
    case Dataflow::Tag::Filter:
      return os << "Filter(" << dataflow.filter._0 << ", " << dataflow.filter._1
                << ")";
    case Dataflow::Tag::Union:
      return os << "Union(" << dataflow.union_._0 << ", " << dataflow.union_._1
                << ")";
    case Dataflow::Tag::Join:
      return os << "Join(" << dataflow.join._0 << ", " << dataflow.join._1
                << ")";
    case Dataflow::Tag::Intersect:
      return os << "Intersect(" << dataflow.intersect._0 << ", "
                << dataflow.intersect._1 << ")";
    case Dataflow::Tag::Product:
      return os << "Product(" << dataflow.product._0 << ", "
                << dataflow.product._1 << ")";
    case Dataflow::Tag::Antijoin:
      return os << "Antijoin(" << dataflow.antijoin._0 << ", "
                << dataflow.antijoin._1 << ")";
    case Dataflow::Tag::Difference:
      return os << "Difference(" << dataflow.difference._0 << ", "
                << dataflow.difference._1 << ")";
    case Dataflow::Tag::OverwriteOne:
      return os << "OverwriteOne(" << dataflow.overwrite_one._0 << ")";
    default:
      throw std::runtime_error("unreachable");
  }
}
