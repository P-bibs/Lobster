use super::ast::*;

use super::grammar;

pub fn parse_item(s: &str) -> Item {
  let parser = grammar::ItemParser::new();
  let result = parser.parse(s);
  result.unwrap()
}

pub fn parse_items(s: &str) -> Vec<Item> {
  let parser = grammar::ItemsParser::new();
  let result = parser.parse(s);
  result.unwrap()
}
