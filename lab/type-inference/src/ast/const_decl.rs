use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub struct _ConstDecl {
  pub name: Ident,
  pub ty: Option<Type>,
  pub value: Expr,
}
