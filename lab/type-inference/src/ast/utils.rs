pub use ast_derive::AstNode;

/// Location of a character (row, column)
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CharLocation {
  pub row: usize,
  pub col: usize,
}

/// A span of two locations (which is parametrized)
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span<T> {
  pub start: T,
  pub end: T,
}

impl<T> Span<T> {
  /// Create a new span
  pub fn new(start: T, end: T) -> Self {
    Self { start, end }
  }
}

impl Span<usize> {
  pub fn is_default(&self) -> bool {
    self.start == 0 && self.end == 0
  }

  pub fn length(&self) -> usize {
    self.end - self.start
  }
}

/// The location of an AST Node
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeLocation {
  pub offset_span: Span<usize>,
  pub loc_span: Option<Span<CharLocation>>,
  pub id: Option<usize>,
  pub source_id: usize,
}

impl std::hash::Hash for NodeLocation {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    state.write_usize(self.source_id);
    state.write_usize(self.offset_span.start);
    state.write_usize(self.offset_span.end);
  }
}

impl NodeLocation {
  /// When cloning a location, we want to keep everything but not the id.
  pub fn clone_without_id(&self) -> Self {
    Self {
      offset_span: self.offset_span.clone(),
      loc_span: self.loc_span.clone(),
      id: None,
      source_id: self.source_id,
    }
  }

  /// Create a location from a single offset span.
  pub fn from_span(start: usize, end: usize) -> Self {
    Self {
      offset_span: Span::new(start, end),
      loc_span: None,
      id: None,
      source_id: 0,
    }
  }
}

impl std::fmt::Debug for NodeLocation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match (&self.id, &self.loc_span) {
      (None, None) => {
        write!(f, "[{}-{}]", self.offset_span.start, self.offset_span.end)
      }
      (Some(id), None) => {
        write!(f, "[#{}, {}-{}]", id, self.offset_span.start, self.offset_span.end)
      }
      (None, Some(loc_span)) => {
        write!(
          f,
          "[{}:{}-{}:{}]",
          loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
      (Some(id), Some(loc_span)) => {
        write!(
          f,
          "[#{}, {}:{}-{}:{}]",
          id, loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
    }
  }
}

/// An AST Node trait
pub trait AstNode: Clone {
  /// Obtain a location of the AstNode
  fn location(&self) -> &NodeLocation;

  /// Obtain a mutable location of the AstNode
  fn location_mut(&mut self) -> &mut NodeLocation;
}

#[derive(Clone, Debug)]
pub struct AstStructNode<T> {
  pub _loc: NodeLocation,
  pub _node: T,
}

impl<T: Clone> AstNode for AstStructNode<T> {
  fn location(&self) -> &NodeLocation {
    &self._loc
  }

  fn location_mut(&mut self) -> &mut NodeLocation {
    &mut self._loc
  }
}

#[derive(Clone, Debug)]
pub struct AstEnumNode<T> {
  pub _loc: NodeLocation,
  pub _node: T,
}

impl<T: Clone> AstNode for AstEnumNode<T> {
  fn location(&self) -> &NodeLocation {
    &self._loc
  }

  fn location_mut(&mut self) -> &mut NodeLocation {
    &mut self._loc
  }
}

pub trait AstNodeVisitor<N> {
  fn visit(&mut self, node: &N);

  fn visit_mut(&mut self, node: &mut N);
}

#[allow(unused)]
impl<V, N> AstNodeVisitor<N> for V {
  default fn visit(&mut self, node: &N) {}

  default fn visit_mut(&mut self, node: &mut N) {}
}

#[allow(unused)]
pub trait AstLocVisitor {
  fn visit_loc(&mut self, loc: &NodeLocation) {}

  fn visit_loc_mut(&mut self, loc: &mut NodeLocation) {}
}

impl<N: AstNode, V: AstLocVisitor> AstNodeVisitor<N> for V {
  fn visit(&mut self, node: &N) {
    self.visit_loc(node.location());
  }

  fn visit_mut(&mut self, node: &mut N) {
    self.visit_loc_mut(node.location_mut());
  }
}

#[allow(unused)]
pub trait AstWalker: Sized {
  fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V);

  fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V);
}

macro_rules! derive_ast_walker {
  ($ty:ty) => {
    impl AstWalker for $ty {
      fn walk<V: AstNodeVisitor<Self>>(&self, _: &mut V) {}
      fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, _: &mut V) {}
    }
  };
}

derive_ast_walker!(i8);
derive_ast_walker!(i16);
derive_ast_walker!(i32);
derive_ast_walker!(i64);
derive_ast_walker!(i128);
derive_ast_walker!(isize);
derive_ast_walker!(u8);
derive_ast_walker!(u16);
derive_ast_walker!(u32);
derive_ast_walker!(u64);
derive_ast_walker!(u128);
derive_ast_walker!(usize);
derive_ast_walker!(f32);
derive_ast_walker!(f64);
derive_ast_walker!(bool);
derive_ast_walker!(char);
derive_ast_walker!(String);

impl<T> AstWalker for Vec<T> where T: AstWalker {
  fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V) {
    for child in self {
      child.walk(v)
    }
  }

  fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V) {
    for child in self {
      child.walk_mut(v)
    }
  }
}

impl<T> AstWalker for Option<T> where T: AstWalker {
  fn walk<V: AstNodeVisitor<Self>>(&self, v: &mut V) {
    if let Some(n) = self {
      n.walk(v)
    }
  }

  fn walk_mut<V: AstNodeVisitor<Self>>(&mut self, v: &mut V) {
    if let Some(n) = self {
      n.walk_mut(v)
    }
  }
}
