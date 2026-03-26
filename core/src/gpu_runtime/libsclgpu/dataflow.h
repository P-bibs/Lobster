#pragma once

#include <thrust/device_vector.h>

#include <ostream>
#include <unordered_map>

#include "bindings.h"
#include "table.h"
#include "table_index.h"

template <typename Prov>
class DynamicRelation {
 private:
  TupleType schema_;

  //std::optional<managed_ptr<HashIndex>> stable_index_;
  Table<Prov> stable_;
  Table<Prov> recent_;
  Table<Prov> delta_;

 public:
  DynamicRelation(const TupleType &schema)
      : schema_(schema), stable_(schema), recent_(schema), delta_(schema) {
    validate();
  }
  DynamicRelation(const DynamicRelation &other) = delete;
  DynamicRelation &operator=(const DynamicRelation &other) = delete;
  DynamicRelation(DynamicRelation &&other) = default;
  DynamicRelation &operator=(DynamicRelation &&other) = default;

  //void set_stable_index(managed_ptr<HashIndex> index) {
  //  stable_index_ = std::move(index);
  //}
  //managed_ptr<HashIndex> &stable_index() {
  //  if (!stable_index_.has_value()) {
  //    throw std::runtime_error("stable index not set");
  //  }
  //  return stable_index_.value();
  //}

  const Table<Prov> &stable() const { return stable_; }
  const Table<Prov> &recent() const { return recent_; }
  const Table<Prov> &delta() const { return delta_; }
  Table<Prov> &stable() { return stable_; }
  Table<Prov> &recent() { return recent_; }
  Table<Prov> &delta() { return delta_; }

  void validate() const {
    stable_.validate();
    recent_.validate();
    delta_.validate();
  }

  const TupleType &schema() const { return schema_; }

  template <typename P>
  friend std::ostream &operator<<(std::ostream &os,
                                  const DynamicRelation<P> &relation);
};

template <typename Prov>
class DynamicIdb {
 private:
  std::unordered_map<std::string, DynamicRelation<Prov>> relations_;

 public:
  DynamicIdb();
  DynamicIdb(
      std::unordered_map<std::string, DynamicRelation<Prov>> &&relations);
  const DynamicRelation<Prov> &operator[](const std::string &name) const;
  DynamicRelation<Prov> &operator[](const std::string &name);
  std::vector<std::string> incorporate_delta(const Prov &ctx, Allocator alloc);
  void validate() const;

  const std::unordered_map<std::string, DynamicRelation<Prov>> &relations()
      const;

  std::unordered_map<std::string, DynamicRelation<Prov>> &relations();

  template <typename P>
  friend std::ostream &operator<<(std::ostream &os, const DynamicIdb<P> &idb);
};

class StratumContext {
 private:
  std::vector<Update> updates_;
};

template <typename Prov>
class ActiveDataflow {
 public:
  virtual TupleType result_schema() const = 0;
  virtual std::vector<std::string> dependencies() const = 0;
  virtual Table<Prov> evaluate_stable(const DynamicIdb<Prov> &idb,
                                      const Prov &ctx) const = 0;
  virtual Table<Prov> evaluate_recent(const DynamicIdb<Prov> &idb,
                                      const Prov &ctx) const = 0;

  static std::unique_ptr<ActiveDataflow> from_dataflow(const Dataflow &df,
                                                       DynamicIdb<Prov> &idb);

  virtual ~ActiveDataflow() {}

  virtual void serialize(std::ostream &os) const = 0;

  friend std::ostream &operator<<(std::ostream &os,
                                  const ActiveDataflow &dataflow) {
    dataflow.serialize(os);
    return os;
  }
};

template <typename Prov>
class ActiveDataflowSource : public ActiveDataflow<Prov> {
 public:
  enum class Op { Relation };

 private:
  Op op_;
  std::string name_;
  TupleType schema_;

 public:
  ActiveDataflowSource(Op op, std::string name, TupleType schema);

  TupleType result_schema() const override;
  std::vector<std::string> dependencies() const override;

  Table<Prov> evaluate_stable(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;
  Table<Prov> evaluate_recent(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;

  void serialize(std::ostream &os) const override;
};

template <typename Prov>
class ActiveDataflowBinaryOp : public ActiveDataflow<Prov> {
 public:
  enum class Op { Join, Product, Difference, Intersection, Union };

 private:
  Op op_;
  bool index_on_right_;
  std::unique_ptr<ActiveDataflow<Prov>> left_;
  std::unique_ptr<ActiveDataflow<Prov>> right_;

 public:
  ActiveDataflowBinaryOp(Op op, std::unique_ptr<ActiveDataflow<Prov>> left,
                         std::unique_ptr<ActiveDataflow<Prov>> right);

  TupleType result_schema() const override;
  std::vector<std::string> dependencies() const override;

  Table<Prov> evaluate_stable(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;
  Table<Prov> evaluate_recent(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;

  bool &index_on_right();

  void serialize(std::ostream &os) const override;
};

template <typename Prov>
class ActiveDataflowUnaryOp : public ActiveDataflow<Prov> {
 public:
  enum class Op { Project, Filter, Find, OverwriteOne };

 private:
  Op op_;
  std::unique_ptr<ActiveDataflow<Prov>> source_;
  Expr expr_;

 public:
  ActiveDataflowUnaryOp(Op op, std::unique_ptr<ActiveDataflow<Prov>> source,
                        Expr expr);

  TupleType result_schema() const override;
  std::vector<std::string> dependencies() const override;

  Table<Prov> evaluate_stable(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;
  Table<Prov> evaluate_recent(const DynamicIdb<Prov> &idb,
                              const Prov &ctx) const override;

  void serialize(std::ostream &os) const override;
};
