use type_inference::parser::parse_items;
use type_inference::analyzers::IdAllocator;
use type_inference::ast::*;

#[test]
fn annotate_id_1() {
  let mut items = parse_items(r#"type List<T> = Nil | Cons(T, List<T>)"#);
  let mut allocator = IdAllocator::new();
  items.walk_mut(&mut allocator);
  println!("{items:#?}");
}

#[test]
fn annotate_id_2() {
  let mut items = parse_items(r#"rel path<usize> = {(0, 1), (1, 2)}"#);
  let mut allocator = IdAllocator::new();
  items.walk_mut(&mut allocator);
  println!("{items:#?}");
}
