use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub enum RelationDecl {
  Fact(FactDecl),
  Set(SetDecl),
  Rule(RuleDecl),
}

#[derive(Clone, Debug, AstNode)]
pub struct _FactDecl {
  pub tag: Option<Tag>,
  pub atom: Atom,
}

#[derive(Clone, Debug, AstNode)]
pub struct _SetDecl {
  pub pred: Predicate,
  pub is_disjunction: bool,
  pub tuples: Vec<SetTuple>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _SetTuple {
  pub tag: Option<Tag>,
  pub args: Vec<Expr>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _RuleDecl {
  pub head: RuleHead,
  pub body: Formula,
}

#[derive(Clone, Debug, AstNode)]
pub enum RuleHead {
  Single(Atom),
  Multi(MultiHead),
}

#[derive(Clone, Debug, AstNode)]
pub struct _MultiHead {
  pub atoms: Vec<Atom>,
  pub is_disjunction: bool,
}
