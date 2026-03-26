use crate::{compiler::ram::Stratum, runtime::provenance::Provenance};

use super::{array::{C_Array, C_String}, dataflow::C_Update, relation::C_Relation, tuple::C_TupleType};

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Stratum<Prov: Provenance> {
  pub relation_names: C_Array<C_String>,
  pub relations: C_Array<C_Relation<Prov>>,
  pub updates: C_Array<C_Update>,
}
impl<Prov: Provenance> std::fmt::Display for C_Stratum<Prov> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Stratum:\n")?;
    write!(f, "  Relations:\n")?;
    write!(f, "    {}\n", self.relation_names)?;
    write!(f, "  Updates:\n")?;
    write!(f, "    {}", self.updates)?;
    Ok(())
  }
}

impl<Prov: Provenance> C_Stratum<Prov> {
  pub fn from_stratum(stratum: &Stratum) -> Self {
    let relations = stratum
      .relations
      .iter()
      .map(|(name, relation)| {
        let predicate = C_String::new(relation.predicate.clone());
        let schema = C_TupleType::from_tuple_type(&relation.tuple_type);
        if !relation.facts.is_empty() {
          panic!("Facts are not supported yet");
        }
        (
          name,
          C_Relation {
            predicate,
            schema,
            size: 0,
            tags: C_Array::empty(),
            tuples: C_Array::empty(),
          },
        )
      })
      .collect::<Vec<_>>();
    let relation_names = relations
      .iter()
      .map(|(name, _)| C_String::new(name.to_string().clone()))
      .collect::<Vec<_>>();
    let relations = relations
      .iter()
      .map(|(_, relation)| relation.clone())
      .collect::<Vec<_>>();
    let relation_names = C_Array::new(relation_names);
    let relations = C_Array::new(relations);

    let updates = stratum
      .updates
      .iter()
      .map(|update| C_Update::from_update(update))
      .collect::<Vec<_>>();
    let updates = C_Array::new(updates);

    Self {
      relation_names,
      relations,
      updates,
    }
  }
}
