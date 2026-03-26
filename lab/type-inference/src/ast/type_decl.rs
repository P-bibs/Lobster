use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub enum TypeDecl {
  Alias(AliasTypeDecl),
  Struct(StructTypeDecl),
  Algebraic(AlgebraicDataTypeDecl),
  Relation(RelationTypeDecl),
}

#[derive(Clone, Debug, AstNode)]
pub struct _AliasTypeDecl {
  pub left: Type,
  pub right: Type,
}

#[derive(Clone, Debug, AstNode)]
pub struct _StructTypeDecl {
  pub name: Type,
  pub fields: Vec<TypeDeclStructField>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _AlgebraicDataTypeDecl {
  pub left: Type,
  pub variants: Vec<AlgebraicDataTypeVariant>,
}

#[derive(Clone, Debug, AstNode)]
pub enum AlgebraicDataTypeVariant {
  Const(Ident),
  Tuple(AlgebraicDataTypeTupleVariant),
  Struct(AlgebraicDataTypeStructVariant),
}

#[derive(Clone, Debug, AstNode)]
pub struct _AlgebraicDataTypeTupleVariant {
  pub constructor: Ident,
  pub arg_types: Vec<Type>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _AlgebraicDataTypeStructVariant {
  pub constructor: Ident,
  pub arg_types: Vec<TypeDeclStructField>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _TypeDeclStructField {
  pub name: Ident,
  pub ty: Type,
}

#[derive(Clone, Debug, AstNode)]
pub struct _RelationTypeDecl {
  pub predicate: Predicate,
  pub params: Vec<Ident>,
  pub arg_types: Vec<RelationTypeArg>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _RelationTypeArg {
  pub adornment: Option<Adornment>,
  pub name: Option<Ident>,
  pub ty: Type,
}

#[derive(Clone, Debug, AstNode)]
pub enum _Adornment {
  Bound,
  Free,
}
