use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub enum Tag {
  Probability(FloatLiteral),
  Boolean(BoolLiteral),
  Natural(IntegerLiteral),
}

#[derive(Clone, Debug, AstNode)]
pub enum _BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Modulo,
  And,
  Or,
  Xor,
  Eq,
  Neq,
  Gt,
  Geq,
  Lt,
  Leq,
}

#[derive(Clone, Debug, AstNode)]
pub enum _UnaryOp {
  Not,
  Pos,
  Neg,
}

#[derive(Clone, Debug, AstNode)]
pub enum Constant {
  Integer(IntegerLiteral),
  Float(FloatLiteral),
  Char(CharLiteral),
  Bool(BoolLiteral),
  String(StringLiteral),
  Symbol(SymbolLiteral),
  DateTime(DateTimeLiteral),
  Duration(DurationLiteral),
}

#[derive(Clone, Debug, AstNode)]
pub struct _IntegerLiteral {
  pub value: i64,
}

#[derive(Clone, Debug, AstNode)]
pub struct _FloatLiteral {
  pub value: f64,
}

#[derive(Clone, Debug, AstNode)]
pub struct _BoolLiteral {
  pub value: bool,
}

#[derive(Clone, Debug, AstNode)]
pub struct _CharLiteral {
  pub value: String,
}

#[derive(Clone, Debug, AstNode)]
pub struct _StringLiteral {
  pub value: String,
}

#[derive(Clone, Debug, AstNode)]
pub struct _SymbolLiteral {
  pub value: String,
}

#[derive(Clone, Debug, AstNode)]
pub struct _DateTimeLiteral {
  pub value: String,
}

#[derive(Clone, Debug, AstNode)]
pub struct _DurationLiteral {
  pub value: String,
}

#[derive(Clone, Debug, AstNode)]
pub struct _Ident {
  pub ident: String,
}

#[derive(Clone, Debug, AstNode)]
pub enum IdentOrWildcard {
  Ident(Ident),
  Wildcard(Wildcard),
}

#[derive(Clone, Debug, AstNode)]
pub struct _Predicate {
  pub name: Ident,
  pub type_params: Vec<Type>,
}
