use crate::compiler::ram::{Update, Dataflow};

use super::{array::C_String, expr::C_Expr, tuple::C_TupleType, value::C_Value};
use crate::common::generic_tuple::GenericTuple;

#[derive(Debug, Clone)]
#[repr(C)]
pub enum C_Dataflow {
  Unit(C_TupleType),
  //UntaggedVec(Vec<Tuple>),
  Relation(C_String),

  Project(*mut C_Dataflow, C_Expr),
  Filter(*mut C_Dataflow, C_Expr),
  Find(*mut C_Dataflow, C_Value),
  Union(*mut C_Dataflow, *mut C_Dataflow),
  /// left, right, index_on_right
  Join(*mut C_Dataflow, *mut C_Dataflow, bool),
  Intersect(*mut C_Dataflow, *mut C_Dataflow),
  Product(*mut C_Dataflow, *mut C_Dataflow),
  Antijoin(*mut C_Dataflow, *mut C_Dataflow),
  Difference(*mut C_Dataflow, *mut C_Dataflow),
  //Reduce(Reduce),

  OverwriteOne(*mut C_Dataflow),
  //Exclusion(Box<C_Dataflow>, Box<C_Dataflow>),

  //ForeignPredicateGround(String, Vec<Value>),
  //ForeignPredicateConstraint(Box<C_Dataflow>, String, Vec<Expr>),
  //ForeignPredicateJoin(Box<C_Dataflow>, String, Vec<Expr>),
}
impl std::fmt::Display for C_Dataflow {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    unsafe {
      match self {
        C_Dataflow::Unit(tuple_type) => write!(f, "{}", tuple_type),
        C_Dataflow::Relation(relation) => write!(f, "{}", relation),
        C_Dataflow::Project(dataflow, expr) => write!(f, "Project({})\n<-{})", expr, **dataflow),
        C_Dataflow::Filter(dataflow, expr) => write!(f, "Filter({})\n<-{})", expr, **dataflow),
        C_Dataflow::Find(dataflow, key) => write!(f, "Find({})\n<-{})", key, **dataflow),
        C_Dataflow::Union(left, right) => write!(f, "Union\n<-{}\n  {})", **left, **right),
        C_Dataflow::Join(left, right, _) => write!(f, "Join\n<-{}\n  {})", **left, **right),
        C_Dataflow::Intersect(left, right) => write!(f, "Intersect\n<-{}\n  {})", **left, **right),
        C_Dataflow::Product(left, right) => write!(f, "Product\n<-{}\n  {})", **left, **right),
        C_Dataflow::Antijoin(left, right) => write!(f, "Antijoin\n<-{}\n  {})", **left, **right),
        C_Dataflow::Difference(left, right) => write!(f, "Difference\n<-{}\n  {})", **left, **right),
        C_Dataflow::OverwriteOne(dataflow) => write!(f, "OverwriteOne\n<-{})", **dataflow),
      }
    }
  }
}
impl C_Dataflow {
  pub fn from_dataflow(dataflow: &Dataflow) -> Self {
    match dataflow {
      Dataflow::Unit(tuple_type) => C_Dataflow::Unit(C_TupleType::from_tuple_type(tuple_type)),
      Dataflow::Relation(relation) => C_Dataflow::Relation(C_String::new(relation.clone())),
      Dataflow::Project(dataflow, expr) => C_Dataflow::Project(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(dataflow))),
        C_Expr::from_expr(expr),
      ),
      Dataflow::Filter(dataflow, expr) => C_Dataflow::Filter(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(dataflow))),
        C_Expr::from_expr(expr),
      ),
      Dataflow::Find(_, GenericTuple::Tuple(_)) => panic!("Find with Tuple not implemented"),
      Dataflow::Find(dataflow, GenericTuple::Value(key)) => C_Dataflow::Find(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(dataflow))),
        C_Value::from_value(key),
      ),
      Dataflow::Union(left, right) => C_Dataflow::Union(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(left))),
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(right))),
      ),
      Dataflow::Join(left, right, index_on_right) => C_Dataflow::Join(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(left))),
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(right))),
        *index_on_right,
      ),
      Dataflow::Intersect(left, right) => C_Dataflow::Intersect(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(left))),
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(right))),
      ),
      Dataflow::Product(left, right) => C_Dataflow::Product(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(left))),
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(right))),
      ),
      Dataflow::OverwriteOne(dataflow) => C_Dataflow::OverwriteOne(
        Box::into_raw(Box::new(C_Dataflow::from_dataflow(dataflow)))),
      _ => panic!("Dataflow not implemented: {:?}", dataflow),
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Update {
  pub target: C_String,
  pub dataflow: C_Dataflow,
}
impl std::fmt::Display for C_Update {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{} <- {}", self.target, self.dataflow)
  }
}
impl C_Update {
  pub fn from_update(update: &Update) -> Self {
    let target = C_String::new(update.target.clone());
    let dataflow = C_Dataflow::from_dataflow(&update.dataflow);
    C_Update { target, dataflow }
  }
}
