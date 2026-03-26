use type_inference::parser::parse_items;

#[test]
fn exp_ti_grammar_1() {
  let result = parse_items(r#"
type edge<T>(x: T, y: T)
rel path<T>(a, b) = edge<T>(a, b) or (path<T>(a, c) and edge<T>(c, b))

rel num_animals(n) = n := count(o: obj(o) and is_animal(o))
rel exists_elephant() = exists(o: obj(o) and is_a(o, "elephant"))
  "#);

  println!("{result:#?}");
}
