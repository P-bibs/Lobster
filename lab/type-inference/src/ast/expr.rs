use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub enum Expr {
  Wildcard(Wildcard),
  Constant(Constant),
  Variable(Ident),
  Entity(Entity),
  New(NewExpr),
  Binary(BinaryExpr),
  Unary(UnaryExpr),
  IfThenElse(IfThenElseExpr),
  Cast(CastExpr),
  Call(CallExpr),
}

#[derive(Clone, Debug, AstNode)]
pub enum Entity {
  Tuple(TupleEntity),
  Struct(StructEntity),
}

#[derive(Clone, Debug, AstNode)]
pub struct _TupleEntity {
  pub constructor: Ident,
  pub args: Vec<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _StructEntity {
  pub constructor: Ident,
  pub args: Vec<StructEntityArg>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _StructEntityArg {
  pub name: Ident,
  pub value: Expr,
}

#[derive(Clone, Debug, AstNode)]
pub struct _BinaryExpr {
  pub op: BinaryOp,
  pub op1: Box<Expr>,
  pub op2: Box<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _UnaryExpr {
  pub op: UnaryOp,
  pub op1: Box<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _CastExpr {
  pub e: Box<Expr>,
  pub ty: Type,
}

#[derive(Clone, Debug, AstNode)]
pub struct _Wildcard;

#[derive(Clone, Debug, AstNode)]
pub struct _IfThenElseExpr {
  pub cond: Box<Expr>,
  pub then_br: Box<Expr>,
  pub else_br: Box<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _NewExpr {
  pub constructor: Ident,
  pub args: Vec<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _CallExpr {
  pub function: Ident,
  pub args: Vec<Expr>,
}
