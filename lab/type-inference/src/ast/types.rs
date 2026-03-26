use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub struct _Type {
  pub name: Ident,
  pub args: Vec<Type>,
}
