use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub struct _Item {
  pub attrs: Vec<Attribute>,
  pub decl: Decl,
}

#[derive(Clone, Debug, AstNode)]
pub enum Decl {
  Import(ImportDecl),
  Type(TypeDecl),
  Const(ConstDecl),
  Relation(RelationDecl),
}
