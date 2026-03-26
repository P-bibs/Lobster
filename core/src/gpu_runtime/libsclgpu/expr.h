#pragma once

#include "bindings.h"
#include "device_vec.h"
#include "table.h"

class BakedPermuteExpr {
 public:
  enum class Kind { Index, Constant };

  struct IndexBody {
    uint64_t index;
  };
  struct ConstantBody {
    Value value;
  };

  struct Body {
    Kind kind;
    union {
      IndexBody index;
      ConstantBody constant;
    };
    Body(IndexBody index) : kind(Kind::Index), index(index) {}
    Body(ConstantBody constant) : kind(Kind::Constant), constant(constant) {}
    Body(const Body &other) : kind(other.kind) {
      if (kind == Kind::Index) {
        index = other.index;
      } else if (kind == Kind::Constant) {
        constant = other.constant;
      } else {
        throw std::runtime_error("invalid expression body kind");
      }
    }
  };

 private:
  std::vector<Body> values_;

  BakedPermuteExpr(const std::vector<Body> &values);

 public:
  static BakedPermuteExpr bake(const TupleType &schema, const Expr &expr);

  template <typename Prov>
  Table<Prov> evaluate(const Table<Prov> &source,
                       const TupleType &result_schema) const;

  size_t size() const;
};

class BakedExpr {
 public:
  static constexpr size_t MAX_EXPR_SIZE = 8;

  enum class Instr { Access, Constant, Binary };

  struct Access {
    Value::Tag type;
    size_t index;
  };
  struct Constant {
    Value value;
  };
  struct Binary {
    BinaryOp op;
  };

  union Body {
    Access access;
    Constant constant;
    Binary binary;

    __host__ __device__ Body() {}
    __host__ __device__ Body(Access access) : access(access) {}
    __host__ __device__ Body(Constant constant) : constant(constant) {}
    __host__ __device__ Body(Binary binary) : binary(binary) {}

    Body(const Body &other) {
      std::copy(reinterpret_cast<const char *>(&other),
                reinterpret_cast<const char *>(&other) + sizeof(Body),
                reinterpret_cast<char *>(this));
    }
    Body &operator=(const Body &other) {
      std::copy(reinterpret_cast<const char *>(&other),
                reinterpret_cast<const char *>(&other) + sizeof(Body),
                reinterpret_cast<char *>(this));
      return *this;
    }
  };

  size_t size_;
  Instr instrs_[MAX_EXPR_SIZE];
  Body bodies_[MAX_EXPR_SIZE];

 private:
  static std::pair<std::vector<BakedExpr::Instr>, std::vector<BakedExpr::Body>>
  compile_expr(const Expr &expr, const TupleType &source_schema);

 public:
  BakedExpr(const Expr &expr, const TupleType &source_schema);

  __host__ __device__ BakedExpr(const BakedExpr &other);
  __host__ __device__ ~BakedExpr();
  __host__ __device__ BakedExpr &operator=(const BakedExpr &other);

  template <typename Prov>
  __device__ static Value eval(const BakedExpr &program,
                               const TableView<Prov> &source, size_t row);

  bool has_division() const;

  friend std::ostream &operator<<(std::ostream &os, const BakedExpr &expr);
};

class BakedTupleExpr {
  std::vector<BakedExpr> exprs_;

 public:
  BakedTupleExpr(const Expr &expr, const TupleType &source_schema);
  size_t size() const { return exprs_.size(); }
  BakedExpr &at(size_t i) { return exprs_[i]; }
  bool has_division() const;

  friend std::ostream &operator<<(std::ostream &os, const BakedTupleExpr &expr);
};
