#include <deque>
#include <string>

#include "bindings.h"
#include "device_vec.h"
#include "expr.h"
#include "flame.h"
#include "normalize.h"
#include "provenance.h"
#include "utils.h"

BakedPermuteExpr::BakedPermuteExpr(const std::vector<Body> &values)
    : values_(values) {}

BakedPermuteExpr BakedPermuteExpr::bake(const TupleType &schema,
                                        const Expr &expr) {
  std::vector<Body> values;

  std::function<void(const Expr &expr)> add_to_values = [&](const Expr &expr) {
    if (expr.tag == Expr::Tag::Tuple) {
      for (size_t i = 0; i < expr.tuple._0.size(); i++) {
        add_to_values(expr.tuple._0[i]);
      }
    } else if (expr.tag == Expr::Tag::Access) {
      values.emplace_back(IndexBody{expr.access._0.to_index(schema)});
    } else if (expr.tag == Expr::Tag::Constant) {
      values.emplace_back(ConstantBody{expr.constant._0});
    }
  };
  add_to_values(expr);
  return BakedPermuteExpr(values);
}

template <typename Prov>
Table<Prov> BakedPermuteExpr::evaluate(const Table<Prov> &source,
                                       const TupleType &result_schema) const {
  TRACE_START(BakedPermuteExpr_evaluate);
  Array<device_buffer> result_facts(values_.size());

  for (size_t i = 0; i < values_.size(); i++) {
    auto &body = values_[i];
    if (body.kind == Kind::Index) {
      auto access_at = body.index.index;
      result_facts[i] = source.column(access_at).clone();
    } else if (body.kind == Kind::Constant) {
      result_facts[i] =
          device_buffer(source.size(), body.constant.value.type());

      DISPATCH_ON_TYPE(body.constant.value.type(), T,
                       thrust::fill(thrust::device, result_facts[i].begin<T>(),
                                    result_facts[i].end<T>(),
                                    body.constant.value.downcast<T>()););
    } else {
      throw std::runtime_error("invalid expression body kind");
    }
  }

  Table<Prov> result(result_schema, std::move(source.tags().clone()),
                     std::move(result_facts),
                     std::move(source.sample_mask().clone()));

  return result;
}

size_t BakedPermuteExpr::size() const { return values_.size(); }

BakedExpr::BakedExpr(const Expr &expr, const TupleType &source_schema) {
  auto [instrs, bodies] = BakedExpr::compile_expr(expr, source_schema);
  if (SAFETY) {
    assert(instrs.size() == bodies.size());
  }
  if (instrs.size() > MAX_EXPR_SIZE) {
    throw std::runtime_error("expression too large");
  }

  size_ = instrs.size();
  std::copy(instrs.begin(), instrs.end(), std::begin(instrs_));
  std::copy(bodies.begin(), bodies.end(), std::begin(bodies_));
}
__host__ __device__ BakedExpr::BakedExpr(const BakedExpr &other) {
  size_ = other.size_;
  memcpy(instrs_, other.instrs_, size_ * sizeof(Instr));
  memcpy(bodies_, other.bodies_, size_ * sizeof(Body));
}
__host__ __device__ BakedExpr &BakedExpr::operator=(const BakedExpr &other) {
  size_ = other.size_;
  memcpy(instrs_, other.instrs_, size_ * sizeof(Instr));
  memcpy(bodies_, other.bodies_, size_ * sizeof(Body));
  return *this;
}
__host__ __device__ BakedExpr::~BakedExpr() {}

std::pair<std::vector<BakedExpr::Instr>, std::vector<BakedExpr::Body>>
BakedExpr::compile_expr(const Expr &expr, const TupleType &source_schema) {
  std::vector<Instr> instrs;
  std::vector<Body> bodies;

  std::function<void(const Expr &)> compile_env = [&](const Expr &expr) {
    if (expr.tag == Expr::Tag::Tuple) {
      throw std::runtime_error("tuple not supported in expression");
    } else if (expr.tag == Expr::Tag::Access) {
      instrs.push_back(Instr::Access);
      Value::Tag type =
          Value::tag_from_type(expr.access._0.to_type(source_schema).tag());
      size_t index = expr.access._0.to_index(source_schema);
      bodies.push_back(BakedExpr::Access{type, index});
    } else if (expr.tag == Expr::Tag::Constant) {
      instrs.push_back(Instr::Constant);
      bodies.push_back(BakedExpr::Constant{expr.constant._0});
    } else if (expr.tag == Expr::Tag::Binary) {
      compile_env(*expr.binary._0.op1);
      compile_env(*expr.binary._0.op2);
      instrs.push_back(Instr::Binary);
      bodies.push_back(BakedExpr::Binary{expr.binary._0.op});
    } else {
      throw std::runtime_error("invalid expression tag");
    }
  };

  compile_env(expr);

  return std::make_pair(instrs, bodies);
}

template <typename Prov>
__device__ Value BakedExpr::eval(const BakedExpr &program,
                                 const TableView<Prov> &source, size_t row) {
  Value stack[8];
  size_t stack_top = 0;

  for (size_t i = 0; i < program.size_; i++) {
    if (SAFETY) {
      if (stack_top >= 8) {
        PANIC("stack overflow");
      }
    }
    auto instr = program.instrs_[i];
    auto &body = program.bodies_[i];
    if (instr == Instr::Access) {
      stack[stack_top++] =
          Value(body.access.type, source.at_raw(row, body.access.index));
    } else if (instr == Instr::Constant) {
      stack[stack_top++] = program.bodies_[i].constant.value;
    } else if (instr == Instr::Binary) {
      if (body.binary.op == BinaryOp::Eq) {
        auto op2 = stack[--stack_top];
        auto op1 = stack[--stack_top];
        stack[stack_top++] = Value(Value::Tag::Bool, op1 == op2);
      } else if (body.binary.op == BinaryOp::Neq) {
        auto op2 = stack[--stack_top];
        auto op1 = stack[--stack_top];
        stack[stack_top++] = Value(Value::Tag::Bool, op1 != op2);
      } else if (body.binary.op == BinaryOp::Gt) {
        auto op2 = stack[--stack_top];
        auto op1 = stack[--stack_top];
        stack[stack_top++] = Value(Value::Tag::Bool, op1 > op2);
      } else if (body.binary.op == BinaryOp::Add) {
        auto op2 = stack[--stack_top];
        auto op1 = stack[--stack_top];
        stack[stack_top++] = Value(op1 + op2);
      } else if (body.binary.op == BinaryOp::Sub) {
        auto op2 = stack[--stack_top];
        auto op1 = stack[--stack_top];
        stack[stack_top++] = Value(op1 - op2);
      } else {
        PANIC("unimplemented binary operator");
      }
    } else {
      PANIC("invalid instruction");
    }
  }

  if (SAFETY) {
    if (stack_top != 1) {
      PANIC("stack not empty");
    }
  }
  return stack[0];
}

std::ostream &operator<<(std::ostream &os, const BakedExpr &expr) {
  std::cout << "BakedExpr{";
  for (size_t i = 0; i < expr.size_; i++) {
    auto instr = expr.instrs_[i];
    auto &body = expr.bodies_[i];
    if (instr == BakedExpr::Instr::Access) {
      os << "Access(" << (int)body.access.type << ", " << body.access.index
         << ")";
    } else if (instr == BakedExpr::Instr::Constant) {
      os << "Constant(" << body.constant.value << ")";
    } else if (instr == BakedExpr::Instr::Binary) {
      os << "Binary(" << (int)body.binary.op << ")";
    } else {
      os << "InvalidInstr";
    }
    os << ", ";
  }
  std::cout << "}";
  return os;
}

BakedTupleExpr::BakedTupleExpr(const Expr &expr,
                               const TupleType &source_schema) {
  std::function<void(const Expr &)> compile_tuple_expr = [&](const Expr &expr) {
    if (expr.tag == Expr::Tag::Tuple) {
      for (size_t i = 0; i < expr.tuple._0.size(); i++) {
        compile_tuple_expr(expr.tuple._0[i]);
      }
    } else {
      exprs_.push_back(BakedExpr(expr, source_schema));
    }
  };
  compile_tuple_expr(expr);
}

bool BakedExpr::has_division() const {
  for (int i = 0; i < size_; i++) {
    if (instrs_[i] == Instr::Binary && bodies_[i].binary.op == BinaryOp::Div) {
      return true;
    }
  }
  return false;
}
bool BakedTupleExpr::has_division() const { 
  for (auto &expr : exprs_) {
    if (expr.has_division()) {
      return true;
    }
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, const BakedTupleExpr &expr) {
  std::cout << "BakedTupleExpr{";
  for (auto &baked : expr.exprs_) {
    os << baked << ", ";
  }
  std::cout << "}";
  return os;
}

#define PROV UnitProvenance
template Table<PROV> BakedPermuteExpr::evaluate(const Table<PROV> &,
                                                const TupleType &) const;
template __device__ Value BakedExpr::eval(const BakedExpr &,
                                          const TableView<PROV> &, size_t);
#undef PROV
#define PROV MinMaxProbProvenance
template Table<PROV> BakedPermuteExpr::evaluate(const Table<PROV> &,
                                                const TupleType &) const;
template __device__ Value BakedExpr::eval(const BakedExpr &,
                                          const TableView<PROV> &, size_t);
#undef PROV
#define PROV DiffMinMaxProbProvenance
template Table<PROV> BakedPermuteExpr::evaluate(const Table<PROV> &,
                                                const TupleType &) const;
template __device__ Value BakedExpr::eval(const BakedExpr &,
                                          const TableView<PROV> &, size_t);
#undef PROV
#define PROV DiffAddMultProbProvenance<>
template Table<PROV> BakedPermuteExpr::evaluate(const Table<PROV> &,
                                                const TupleType &) const;
template __device__ Value BakedExpr::eval(const BakedExpr &,
                                          const TableView<PROV> &, size_t);
#undef PROV
#define PROV DiffTopKProofsProvenance<>
template Table<PROV> BakedPermuteExpr::evaluate(const Table<PROV> &,
                                                const TupleType &) const;
template __device__ Value BakedExpr::eval(const BakedExpr &,
                                          const TableView<PROV> &, size_t);
#undef PROV
