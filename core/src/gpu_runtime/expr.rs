use crate::common::{binary_op::BinaryOp, expr::{Expr, BinaryExpr}, tuple_access::{TUPLE_ACCESSOR_DEPTH, TupleAccessor}};

use super::{value::{C_Value, C_ValueType}, array::C_Array};

#[derive(Debug, Clone)]
#[repr(C)]
pub enum C_BinaryOp {
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
}
impl C_BinaryOp {
  pub fn from_binary_op(binary_op: &BinaryOp) -> Self {
    match binary_op {
      BinaryOp::Add => C_BinaryOp::Add,
      BinaryOp::Sub => C_BinaryOp::Sub,
      BinaryOp::Mul => C_BinaryOp::Mul,
      BinaryOp::Div => C_BinaryOp::Div,
      BinaryOp::Mod => C_BinaryOp::Mod,
      BinaryOp::And => C_BinaryOp::And,
      BinaryOp::Or => C_BinaryOp::Or,
      BinaryOp::Xor => C_BinaryOp::Xor,
      BinaryOp::Eq => C_BinaryOp::Eq,
      BinaryOp::Neq => C_BinaryOp::Neq,
      BinaryOp::Lt => C_BinaryOp::Lt,
      BinaryOp::Leq => C_BinaryOp::Leq,
      BinaryOp::Gt => C_BinaryOp::Gt,
      BinaryOp::Geq => C_BinaryOp::Geq,
    }
  }
}
impl std::fmt::Display for C_BinaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      C_BinaryOp::Add => write!(f, "+"),
      C_BinaryOp::Sub => write!(f, "-"),
      C_BinaryOp::Mul => write!(f, "*"),
      C_BinaryOp::Div => write!(f, "/"),
      C_BinaryOp::Mod => write!(f, "%"),
      C_BinaryOp::And => write!(f, "&"),
      C_BinaryOp::Or => write!(f, "|"),
      C_BinaryOp::Xor => write!(f, "^"),
      C_BinaryOp::Eq => write!(f, "=="),
      C_BinaryOp::Neq => write!(f, "!="),
      C_BinaryOp::Lt => write!(f, "<"),
      C_BinaryOp::Leq => write!(f, "<="),
      C_BinaryOp::Gt => write!(f, ">"),
      C_BinaryOp::Geq => write!(f, ">="),
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub enum UnaryOp {
  Neg,
  Pos,
  Not,
  TypeCast(C_ValueType),
}
impl std::fmt::Display for UnaryOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      UnaryOp::Neg => write!(f, "-"),
      UnaryOp::Pos => write!(f, "+"),
      UnaryOp::Not => write!(f, "!"),
      UnaryOp::TypeCast(value_type) => write!(f, "({})", value_type),
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub enum C_Expr {
  Tuple(C_Array<C_Expr>),
  Access(C_TupleAccessor),
  Constant(C_Value),
  Binary(C_BinaryExpr),
  // Unary(UnaryExpr),
  // IfThenElse(IfThenElseExpr),
  // Call(CallExpr),
  // New(NewExpr),
}
impl std::fmt::Display for C_Expr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      C_Expr::Tuple(exprs) => {
        write!(f, "Tuple({})", exprs)
      },
      C_Expr::Access(indices) => {
        write!(f, "{}", indices)
      },
      C_Expr::Constant(value) => {
        write!(f, "{}", value)
      },
      C_Expr::Binary(binary_expr) => {
        write!(f, "{}", binary_expr)
      },
    }
  }
}

impl C_Expr {
  pub fn from_expr(expr: &Expr) -> Self {
    match expr {
      Expr::Tuple(exprs) => {
        let exprs = exprs.iter().map(|expr| C_Expr::from_expr(expr)).collect::<Vec<_>>();
        C_Expr::Tuple(C_Array::new(exprs))
      },
      Expr::Access(indices) => C_Expr::Access(C_TupleAccessor::from_tuple_accessor(indices)),
      Expr::Constant(value) => C_Expr::Constant(C_Value::from_value(value)),
      Expr::Binary(binary_expr) => C_Expr::Binary(C_BinaryExpr::from_binary_expr(binary_expr)),
      _ => unimplemented!(),
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_TupleAccessor {
  len: i8,
  indices: [i8; TUPLE_ACCESSOR_DEPTH],
}
impl std::fmt::Display for C_TupleAccessor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "(")?;
    for i in 0..self.len {
      write!(f, "{}", self.indices[i as usize])?;
      if i < self.len - 1 {
        write!(f, ", ")?;
      }
    }
    write!(f, ")")
  }
}
impl C_TupleAccessor {
  pub fn from_tuple_accessor(tuple_accessor: &TupleAccessor) -> Self {
    let len = tuple_accessor.len() as i8;
    let indices = tuple_accessor.indices.clone();

    Self { len, indices }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_BinaryExpr {
  op: C_BinaryOp,
  op1: *const C_Expr,
  op2: *const C_Expr,
}
impl std::fmt::Display for C_BinaryExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "({:?}, {}, {})", self.op, unsafe { &*self.op1 }, unsafe { &*self.op2 })
  }
}
impl C_BinaryExpr {
  pub fn from_binary_expr(binary_expr: &BinaryExpr) -> Self {
    let op = C_BinaryOp::from_binary_op(&binary_expr.op);
    let op1 = Box::into_raw(Box::new(C_Expr::from_expr(&binary_expr.op1)));
    let op2 = Box::into_raw(Box::new(C_Expr::from_expr(&binary_expr.op2)));

    Self { op, op1, op2 }
  }
}

