#![feature(min_specialization)]

#[derive(Default, Debug)]
pub struct NodeLocation {
  pub id: usize,
}

#[derive(Debug)]
pub struct NodeWrapper<T> {
  pub _node: T,
  pub _loc: NodeLocation,
}

impl<T> NodeWrapper<T> {
  fn new(t: T) -> Self {
    Self {
      _node: t,
      _loc: NodeLocation::default(),
    }
  }
}

// ================= Nodes =================

#[derive(Debug)]
pub struct _Var { pub name: String }
pub type Var = NodeWrapper<_Var>;

#[derive(Debug)]
pub struct _Const { pub val: i32 }
pub type Const = NodeWrapper<_Const>;

#[derive(Debug)]
pub enum _Op { Add, Sub }
pub type Op = NodeWrapper<_Op>;

#[derive(Debug)]
pub struct _BinExpr { op: Op, left: Box<Expr>, right: Box<Expr> }
pub type BinExpr = NodeWrapper<_BinExpr>;

#[derive(Debug)]
pub enum Expr { Var(Var), Const(Const), Binary(BinExpr) }

// ================= Node Walker =================

pub trait NodeWalker: Sized {
  fn walk<V>(&mut self, v: &mut V);
}

impl NodeWalker for Var {
  fn walk<V>(&mut self, v: &mut V) {
    v.visit(self);
    v.visit(&mut self._loc);
  }
}

impl NodeWalker for Const {
  fn walk<V>(&mut self, v: &mut V) {
    v.visit(self);
    v.visit(&mut self._loc);
  }
}

impl NodeWalker for Op {
  fn walk<V>(&mut self, v: &mut V) {
    v.visit(self);
    v.visit(&mut self._loc);
  }
}

impl NodeWalker for BinExpr {
  fn walk<V>(&mut self, v: &mut V) {
    v.visit(self);
    v.visit(&mut self._loc);
    self._node.op.walk(v);
    self._node.left.walk(v);
    self._node.right.walk(v);
  }
}

impl NodeWalker for Expr {
  fn walk<V>(&mut self, v: &mut V) {
    match self {
      Expr::Var(e) => e.walk(v),
      Expr::Const(e) => e.walk(v),
      Expr::Binary(e) => e.walk(v),
    }
  }
}

// ================= Node Visitor =================

pub trait NodeVisitor<N: Sized> {
  fn visit(&mut self, n: &mut N);
}

impl<T, U> NodeVisitor<T> for U {
  default fn visit(&mut self, _n: &mut T) {}
}

impl<T, A, B> NodeVisitor<T> for (&mut A, &mut B) {
  default fn visit(&mut self, n: &mut T) {
    let (a, b) = self;
    <A as NodeVisitor<T>>::visit(a, n);
    <B as NodeVisitor<T>>::visit(b, n);
  }
}

pub struct PrintConst;

impl NodeVisitor<Const> for PrintConst {
  fn visit(&mut self, c: &mut Const) {
    println!("constant: {}", c._node.val)
  }
}

pub struct IdAllocator {
  curr_id: usize,
}

impl NodeVisitor<NodeLocation> for IdAllocator {
  fn visit(&mut self, l: &mut NodeLocation) {
    l.id = self.curr_id;
    self.curr_id += 1;
  }
}

// ================= Main =================

fn main() {
  let mut ast = Expr::Binary(
    BinExpr::new(_BinExpr {
      op: Op::new(_Op::Add),
      left: Box::new(Expr::Var(Var::new(_Var { name: "x".to_string() }))),
      right: Box::new(Expr::Const(Const::new(_Const { val: 1 }))),
    })
  );

  println!("{ast:#?}");

  let mut const_printer = PrintConst;
  let mut id_alloc = IdAllocator { curr_id: 1 };
  let mut transformers = (&mut const_printer, &mut id_alloc);

  ast.walk(&mut transformers);

  println!("{ast:#?}");
}
