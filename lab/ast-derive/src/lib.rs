use proc_macro::*;

/// Abstract Syntax Tree (AST) Node
///
/// An AstNode can be one of two things: a struct or an enum
///
/// A struct AST node must have a name started with underscore (`_`).
#[proc_macro_derive(AstNode)]
pub fn derive_ast_node(tokens: TokenStream) -> TokenStream {
  let token_list: Vec<_> = tokens.into_iter().collect();

  // Get pub and check if the decorated item is a struct
  let has_pub = get_has_pub(&token_list);
  let is_struct = get_is_struct(has_pub, &token_list);

  // Get information
  let pub_kw = if has_pub { "pub" } else { "" };
  let name = get_type_name(has_pub, is_struct, &token_list);

  // Derive!
  let derive = if is_struct {
    derive_struct(pub_kw, name, has_pub, &token_list)
  } else {
    derive_enum(pub_kw, name, has_pub, &token_list)
  };

  derive.parse().unwrap()
}

fn get_has_pub(token_list: &Vec<TokenTree>) -> bool {
  match &token_list[0] {
    TokenTree::Ident(i) => if i.to_string() == "pub" {
      true
    } else {
      false
    },
    _ => false,
  }
}

fn get_is_struct(has_pub: bool, token_list: &Vec<TokenTree>) -> bool {
  let offset = if has_pub { 1 } else { 0 };
  match &token_list[offset] {
    TokenTree::Ident(i) => {
      if i.to_string() == "struct" {
        match &token_list[offset + 2] {
          TokenTree::Group(g) => match g.delimiter() {
            Delimiter::Brace => {
              true
            }
            _ => panic!("AstNode only support decorating struct with fields")
          }
          TokenTree::Punct(p) if p.as_char() == ';' => true,
          t => panic!("Unknown token tree {}", t),
        }
      } else if i.to_string() == "enum" {
        false
      } else {
        panic!("AstNode can only be used to decorate struct or enum")
      }
    }
    t => panic!("Unknown token {}", t),
  }
}

fn get_type_name(has_pub: bool, is_struct: bool, token_list: &Vec<TokenTree>) -> String {
  let offset = if has_pub { 1 } else { 0 };
  match &token_list[offset + 1] {
    TokenTree::Ident(i) => {
      let full_name = i.to_string();
      if is_struct {
        if full_name.chars().nth(0) == Some('_') {
          full_name[1..].to_string()
        } else {
          panic!("Expected struct name to begin with underscore")
        }
      } else {
        full_name
      }
    },
    other => panic!("Unexpected token tree {:?}", other),
  }
}

fn get_struct_fields(has_pub: bool, token_list: &Vec<TokenTree>) -> Vec<(bool, String, String)> {
  let offset = if has_pub { 1 } else { 0 };
  match &token_list[offset + 2] {
    TokenTree::Group(g) => {
      let fields_token_list: Vec<_> = g.stream().into_iter().collect();
      let mut fields = Vec::new();
      let mut ty_indent = 0;
      let mut curr_is_pub = None;
      let mut curr_field_name = None;
      let mut curr_field_type = None;
      for token in fields_token_list {
        if curr_is_pub.is_none() {
          match token {
            TokenTree::Ident(i) => {
              if i.to_string() == "pub" {
                curr_is_pub = Some(true);
              } else {
                curr_is_pub = Some(false);
                curr_field_name = Some(i.to_string());
              }
            },
            _ => panic!("Cannot parse")
          }
        } else if curr_is_pub.is_some() && curr_field_name.is_none() {
          match token {
            TokenTree::Ident(i) => {
              curr_field_name = Some(i.to_string());
            }
            _ => panic!("Cannot parse")
          }
        } else if curr_field_name.is_some() && curr_field_type.is_none() {
          match token {
            TokenTree::Punct(p) if p.as_char() == ':' => {
              curr_field_type = Some("".to_string())
            }
            _ => panic!("Cannot parse")
          }
        } else if curr_field_type.is_some() {
          match token {
            TokenTree::Ident(i) => {
              if let Some(field_type) = &mut curr_field_type {
                *field_type += &i.to_string();
              } else {
                panic!("Cannot parse")
              }
            }
            TokenTree::Group(g) => {
              if let Some(field_type) = &mut curr_field_type {
                *field_type += &g.to_string();
              } else {
                panic!("Cannot parse")
              }
            }
            TokenTree::Punct(p) => {
              if p.as_char() == ',' && ty_indent == 0 {
                  fields.push((curr_is_pub.unwrap(), curr_field_name.unwrap(), curr_field_type.unwrap()));
                  curr_is_pub = None;
                  curr_field_name = None;
                  curr_field_type = None;
              } else {
                if p.as_char() == '<' {
                  ty_indent += 1;
                } else if p.as_char() == '>' {
                  ty_indent -= 1;
                }

                if let Some(field_type) = &mut curr_field_type {
                  *field_type += &p.as_char().to_string();
                } else {
                  panic!("Cannot parse")
                }
              }
            }
            _ => panic!("Cannot parse")
          }
        }
      }

      match (curr_is_pub, curr_field_name, curr_field_type) {
        (Some(is_pub), Some(field_name), Some(field_type)) => {
          fields.push((is_pub, field_name, field_type));
        }
        _ => {}
      };

      fields
    }
    TokenTree::Punct(_) => vec![],
    _ => panic!("Not a group")
  }
}

fn get_enum_variants(has_pub: bool, token_list: &Vec<TokenTree>) -> Vec<(String, Option<String>)> {
  let offset = if has_pub { 1 } else { 0 };
  match &token_list[offset + 2] {
    TokenTree::Group(g) => {
      let variants_token_list: Vec<_> = g.stream().into_iter().collect();
      if !variants_token_list.is_empty() {
        let mut variants = vec![];
        let mut curr_variant_name = None;
        let mut curr_variant_type = None;

        for token in variants_token_list {
          if curr_variant_name.is_none() {
            match token {
              TokenTree::Ident(i) => {
                curr_variant_name = Some(i.to_string());
              }
              _ => panic!("Cannot parse")
            }
          } else if curr_variant_name.is_some() && curr_variant_type.is_none() {
            match token {
              TokenTree::Group(g) => {
                if g.delimiter() == Delimiter::Parenthesis {
                  curr_variant_type = Some(Some(g.stream().into_iter().map(|t| t.to_string()).collect::<Vec<_>>().join("")));
                } else {
                  panic!("AstNode enum variant cannot be a struct")
                }
              }
              TokenTree::Punct(p) if p.as_char() == ',' => {
                variants.push((curr_variant_name.unwrap(), None));
                curr_variant_name = None;
                curr_variant_type = None;
              }
              _ => panic!("Cannot parse")
            }
          } else if curr_variant_name.is_some() && curr_variant_type.is_some() {
            match token {
              TokenTree::Punct(p) if p.as_char() == ',' => {
                variants.push((curr_variant_name.unwrap(), curr_variant_type.unwrap()));
                curr_variant_name = None;
                curr_variant_type = None;
              }
              _ => panic!("Cannot parse")
            }
          } else {
            panic!("Cannot parse")
          }
        }

        match (curr_variant_name, curr_variant_type) {
          (Some(name), Some(ty)) => {
            variants.push((name, ty));
          }
          _ => {}
        };

        variants
      } else {
        panic!("AstNode enum has to have at least one variant")
      }
    }
    _ => panic!("Not a group")
  }
}

fn derive_struct(pub_kw: &str, name: String, has_pub: bool, token_list: &Vec<TokenTree>) -> String {
  let struct_def = format!(r#"
    {pub_kw} type {name} = AstStructNode<_{name}>;
  "#);

  let fields = get_struct_fields(has_pub, &token_list);

  // New and default functions
  let fn_args = fields.iter().map(|(_, name, ty)| format!("{name}: {ty}")).collect::<Vec<_>>().join(",");
  let constructor_args = fields.iter().map(|(_, name, _)| name).cloned().collect::<Vec<_>>().join(",");
  let new_impl = format!(r#"
    impl _{name} {{
      pub fn new({fn_args}) -> Self {{
        Self {{ {constructor_args} }}
      }}
      pub fn with_location(self, loc: NodeLocation) -> {name} {{
        {name} {{
          _loc: loc,
          _node: self,
        }}
      }}
      pub fn with_span(self, start: usize, end: usize) -> {name} {{
        {name} {{
          _loc: NodeLocation::from_span(start, end),
          _node: self,
        }}
      }}
    }}
    impl {name} {{
      pub fn new(loc: NodeLocation, {fn_args}) -> Self {{
        Self {{
          _loc: loc,
          _node: _{name} {{
            {constructor_args}
          }},
        }}
      }}
      pub fn new_without_location({fn_args}) -> Self {{
        Self {{
          _loc: NodeLocation::default(),
          _node: _{name} {{
            {constructor_args}
          }}
        }}
      }}
      pub fn with_node_and_location(node: _{name}, loc: NodeLocation) -> Self {{
        Self {{
          _loc: loc,
          _node: node,
        }}
      }}
      pub fn with_node_and_span(node: _{name}, start: usize, end: usize) -> Self {{
        Self {{
          _loc: NodeLocation::from_span(start, end),
          _node: node,
        }}
      }}
    }}
  "#);

  // Walker
  let (walker, walker_mut): (String, String) = fields
    .iter()
    .map(|(_, field_name, _)| {
      let walk = format!("self._node.{field_name}.walk(v);");
      let walk_mut = format!("self._node.{field_name}.walk_mut(v);");
      (walk, walk_mut)
    })
    .unzip();
  let impl_walker = format!(r#"
    impl AstWalker for {name} {{
      fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V) {{
        v.visit(self);
        {walker}
      }}
      fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V) {{
        v.visit_mut(self);
        {walker_mut}
      }}
    }}
  "#);

  // Accessor
  let fields_accessor = fields
    .iter()
    .map(|(is_pub, field_name, field_type)| {
      let pub_kw = if *is_pub { "pub" } else { "" };
      format!(r#"
        {pub_kw} fn {field_name}(&self) -> &{field_type} {{
          &self._node.{field_name}
        }}
        {pub_kw} fn {field_name}_mut(&mut self) -> &mut {field_type} {{
          &mut self._node.{field_name}
        }}
      "#)
    })
    .collect::<Vec<_>>()
    .join("\n");
  let fields_impl = format!(r#"
    impl {name} {{
      {fields_accessor}
    }}
  "#);

  vec![
    struct_def,
    new_impl,
    impl_walker,
    fields_impl,
  ].join("\n")
}

fn derive_enum(pub_kw: &str, name: String, has_pub: bool, token_list: &Vec<TokenTree>) -> String {
  let variants = get_enum_variants(has_pub, &token_list);
  if variants.iter().all(|(_, ty)| ty.is_some()) {
    derive_variant_enum(name, variants)
  } else if variants.iter().all(|(_, ty)| ty.is_none()) {
    derive_const_enum(pub_kw, name, variants)
  } else {
    panic!("AstNode enum either all have one argument or all have no argument")
  }
}

fn derive_variant_enum(name: String, variants: Vec<(String, Option<String>)>) -> String {
  let imut_cases = variants.iter().map(|(name, _)| format!("Self::{name}(v) => v.location()")).collect::<Vec<_>>().join(",");
  let mut_cases = variants.iter().map(|(name, _)| format!("Self::{name}(v) => v.location_mut()")).collect::<Vec<_>>().join(",");

  let ast_node_impl = format!(r#"
    impl AstNode for {name} {{
      fn location(&self) -> &NodeLocation {{
        match self {{
          {imut_cases}
        }}
      }}
      fn location_mut(&mut self) -> &mut NodeLocation {{
        match self {{
          {mut_cases}
        }}
      }}
    }}
  "#);

  let (walkers, walker_muts): (String, String) = variants
    .iter()
    .map(|(name, _)| {
      (
        format!("Self::{name}(c) => c.walk(v),"),
        format!("Self::{name}(c) => c.walk_mut(v),"),
      )
    })
    .unzip();
  let impl_walker = format!(r#"
    impl AstWalker for {name} {{
      fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V) {{
        match self {{
          {walkers}
        }}
      }}
      fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V) {{
        match self {{
          {walker_muts}
        }}
      }}
    }}
  "#);

  vec![
    ast_node_impl,
    impl_walker,
  ].join("\n")
}

fn derive_const_enum(pub_kw: &str, name: String, variants: Vec<(String, Option<String>)>) -> String {
  assert!(name.chars().nth(0) == Some('_'), "The first character of the name needs to be an underscore `_`");
  let name = name[1..].to_string();
  let type_def = format!(r#"
    {pub_kw} type {name} = AstEnumNode<_{name}>;
  "#);

  let universal_helper = format!(r#"
    impl _{name} {{
      pub fn with_location(self, loc: NodeLocation) -> {name} {{
        {name} {{
          _loc: loc,
          _node: self,
        }}
      }}
      pub fn with_span(self, start: usize, end: usize) -> {name} {{
        {name} {{
          _loc: NodeLocation::from_span(start, end),
          _node: self,
        }}
      }}
    }}
  "#);

  let helpers = variants
    .iter()
    .map(|(variant_name, _)| {
      let lower_case_variant_name = variant_name.to_lowercase();
      format!(r#"
        pub fn {lower_case_variant_name}() -> Self {{
          Self {{
            _loc: NodeLocation::default(),
            _node: _{name}::{variant_name},
          }}
        }}
        pub fn {lower_case_variant_name}_with_loc(loc: NodeLocation) -> Self {{
          Self {{
            _loc: loc,
            _node: _{name}::{variant_name},
          }}
        }}
        pub fn {lower_case_variant_name}_with_span(start: usize, end: usize) -> Self {{
          Self {{
            _loc: NodeLocation::from_span(start, end),
            _node: _{name}::{variant_name},
          }}
        }}
        pub fn is_{lower_case_variant_name}(&self) -> bool {{
          match &self._node {{
            _{name}::{variant_name} => true,
            _ => false,
          }}
        }}
      "#)
    })
    .collect::<Vec<_>>()
    .join("\n");
  let match_helper_impl = format!(r#" impl {name} {{ {helpers} }} "#);

  let impl_walker = format!(r#"
    impl AstWalker for {name} {{
      fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V) {{
        v.visit(self);
      }}
      fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V) {{
        v.visit_mut(self);
      }}
    }}
  "#);

  vec![
    type_def,
    universal_helper,
    impl_walker,
    match_helper_impl,
  ].join("\n")
}
