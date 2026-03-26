use crate::ast::*;

#[derive(Clone, Debug)]
pub struct IdAllocator {
  curr_id: usize,
}

impl IdAllocator {
  pub fn new() -> Self {
    Self {
      curr_id: 0,
    }
  }
}

impl AstLocVisitor for IdAllocator {
  fn visit_loc_mut(&mut self, loc: &mut NodeLocation) {
    let new_id = self.curr_id;
    self.curr_id += 1;
    loc.id = Some(new_id);
  }
}
