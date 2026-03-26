use ast_derive::*;

use super::*;

#[derive(Clone, Debug, AstNode)]
pub struct _Attribute {
  pub name: Ident,
  pub args: Vec<AttributeArg>,
}

#[derive(Clone, Debug, AstNode)]
pub enum AttributeArg {
  Pos(AttributeArgValue),
  Kw(AttributeNamedArgValue),
}

#[derive(Clone, Debug, AstNode)]
pub struct _AttributeNamedArgValue {
  pub kw: Ident,
  pub value: AttributeArgValue,
}

#[derive(Clone, Debug, AstNode)]
pub enum AttributeArgValue {
  Constant(Constant),
  List(AttributeArgValueList),
  Tuple(AttributeArgValueTuple),
  Dict(AttributeArgValueDict),
}

#[derive(Clone, Debug, AstNode)]
pub struct _AttributeArgValueList {
  pub values: Vec<AttributeArgValue>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _AttributeArgValueTuple {
  pub values: Vec<AttributeArgValue>,
}

#[derive(Clone, Debug, AstNode)]
pub struct _AttributeArgValueDict {
  pub values: Vec<AttributeNamedArgValue>,
}
