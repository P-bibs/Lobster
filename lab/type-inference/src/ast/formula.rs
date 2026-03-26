use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub enum Formula {
  Atom(Atom),
  CaseIs(CaseIs),
  Unary(UnaryFormula),
  Binary(BinaryFormula),
  Constraint(Expr),
  Aggregate(Aggregate),
  Range(Range),
}

#[derive(Clone, Debug, AstNode)]
pub struct _Atom {
  pub pred: Predicate,
  pub args: Vec<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _CaseIs {
  pub var: Ident,
  pub cases: Vec<Entity>,
}

#[derive(Clone, Debug, AstNode)]
pub enum _UnaryFormulaOp {
  Not,
}

#[derive(Clone, Debug, AstNode)]
pub struct _UnaryFormula {
  pub op: UnaryFormulaOp,
  pub op1: Box<Formula>,
}

#[derive(Clone, Debug, AstNode)]
pub enum _BinaryFormulaOp {
  And,
  Or,
  Implies,
}

#[derive(Clone, Debug, AstNode)]
pub struct _BinaryFormula {
  pub op: BinaryFormulaOp,
  pub left: Box<Formula>,
  pub right: Box<Formula>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _Aggregate {
  pub left: Vec<Ident>,
  pub aggregator: Aggregator,
  pub body: BindingFormula,
  pub where_body: Option<BindingFormula>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _Aggregator {
  pub aggregator: Ident,
  pub type_params: Vec<Type>,
  pub bracket_args: Vec<Expr>,
  pub exclamation: bool,
}

#[derive(Clone, Debug, AstNode)]
pub struct _BindingFormula {
  pub binding_var_groups: Vec<BindingVarGroup>,
  pub formula: Box<Formula>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _BindingVarGroup {
  pub vars: Vec<IdentOrWildcard>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _Range {
  pub var: Ident,
  pub begin: Box<Expr>,
  pub step: Box<Option<Expr>>,
  pub end: Box<Expr>,
  pub ty: Option<Type>,
}
