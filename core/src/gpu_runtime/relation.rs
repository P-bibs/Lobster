use std::collections::HashMap;
use std::collections::BTreeMap;
use crate::{
  common::{tuple_type::TupleType},
  runtime::{
    database::{
      extensional::ExtensionalDatabase, intentional::{IntentionalDatabase, IntentionalRelation}
    }, dynamic::DynamicCollection, provenance::{Provenance, Tagged}
  }, utils::PointerFamily
};
use rayon::prelude::*;

use super::{tuple::{C_TupleType, value_list_to_tuple}, value::{C_Value}, array::{C_Array, C_String}, tag::C_Tag};


#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Relation<Prov: Provenance> {
  pub predicate: C_String,
  pub schema: C_TupleType,
  pub size: usize,
  pub tags: C_Array<C_Tag<Prov>>,
  pub tuples: C_Array<C_Array<C_Value>>
}
impl<Prov: Provenance> std::fmt::Display for C_Relation<Prov> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}({})\n", self.predicate, self.schema)?;
    write!(f, " {}x{}: {{\n", self.tuples.length(), self.size)?;
    for col in 0..self.tuples.length() {
      for row in 0..self.size {
        write!(f, "{},", self.tuples.get(col).unwrap().get(row).unwrap())?;
      }
      write!(f, "\n")?;
    }
    write!(f, "}}\n")
  }
}
impl<P: Provenance> C_Relation<P> {
  fn width(&self) -> usize {
    self.tuples.length()
  }
}
use crate::common;
use crate::runtime;
type MyType = usize;//Vec<(GenericTuple<common::value::Value>, <Prov as runtime::provenance::provenance::Provenance>::Tag)>;

const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_par<T: IntoParallelIterator>() {}

    // RFC 2056
    fn assert_all() {
        assert_send::<MyType>();
        assert_sync::<MyType>();
    }
};
use crate::common::generic_tuple::GenericTuple;
fn facts_from_iterator<'a, Prov: Provenance>(
  iter: impl Iterator<Item = (String, &'a DynamicCollection<Prov>)>,
  relation_types: &HashMap<String, TupleType>)
-> C_Array<C_Relation<Prov>> {
  let _guard = flame::start_guard("facts_from_iterator");
  let mut relations = Vec::new();
  for (name, collection) in iter {
    let _guard1 = flame::start_guard("map_relation");
    let size = collection.elements().len();
    let mut tuples = Vec::new();
    tuples.reserve(size);
    let mut tags = Vec::new();
    tags.reserve(size);

    flame::start("collect_elements");
    let elements = match collection {
      | DynamicCollection::Sorted(e) => &e.elements,
      | _ => panic!("Unsupported collection type")
    };

    elements.iter().for_each(|Tagged {tuple, tag}| {
      tuples.push(tuple);
      tags.push(tag);
    });
    flame::end("collect_elements");

    let schema = C_TupleType::from_tuple_type(relation_types.get(&name).unwrap());

    let tuple_width = schema.width();

    flame::start("create_tuples");
    let tuples = (0..tuple_width).map(|i| {
      let values = tuples.par_iter().map(|tuple| C_Value::from_value(&tuple.to_values()[i])).collect::<Vec<_>>();
      C_Array::new(values)
    }).collect::<Vec<_>>();
    let tuples = C_Array::new(tuples);
    flame::end("create_tuples");

    // TODO: fix
    flame::start("create_tags");
    let tags = C_Array::new(tags.into_iter().map(|tag| {
      C_Tag::new(tag)
    }).collect::<Vec<_>>());
    flame::end("create_tags");

    flame::start("create_relation");
    let rel = C_Relation { predicate: C_String::new(name.clone()), schema, size, tags, tuples };
    flame::end("create_relation");

    relations.push(rel);
  }
  let relations = C_Array::new(relations);
  return relations;
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct IDB<Prov: Provenance> {
  pub relations: C_Array<C_Relation<Prov>>
}
impl<Prov: Provenance> std::fmt::Display for IDB<Prov> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.relations)
  }
}

impl<Prov: Provenance> IDB<Prov> {
  pub fn from_intentional_database<Ptr: PointerFamily>(idb: &IntentionalDatabase<Prov, Ptr>, relation_types: &HashMap<String, TupleType>) -> Self {
    let iter = idb.intentional_relations.iter().map(|(name, value)| (name.clone(), &value.internal_facts));
    Self { relations: facts_from_iterator(iter, relation_types) }
  }
  pub fn from_intentional_database_with_filter<Ptr: PointerFamily>(idb: &IntentionalDatabase<Prov, Ptr>, relation_types: &HashMap<String, TupleType>, filter: &Vec<String>) -> Self {
    let iter = idb.intentional_relations.iter().filter(|(name, _)| filter.contains(name)).map(|(name, value)| (name.clone(), &value.internal_facts));
    Self { relations: facts_from_iterator(iter, relation_types) }
  }

  pub fn to_intentional_database<Ptr: PointerFamily>(&self, ctx: &Prov) -> IntentionalDatabase<Prov, Ptr> {
    let mut relations = BTreeMap::new();

    for i in 0..(self.relations.length()) {
      let relation = self.relations.get(i).unwrap();

      let name = relation.predicate.to_string();

      let relation_size = relation.size;
      let relation_width = relation.width();

      let facts = (0..relation_size).map(|row| {
        let values = (0..relation_width)
            .map(|col| relation.tuples.get(col).unwrap().get(row).unwrap().clone())
            .collect::<Vec<_>>();

        let ffi_tag = relation.tags.get(row).unwrap();
        let tag = Prov::from_ffi_tag(&ffi_tag.tag);
        Tagged {
          tuple: value_list_to_tuple(values, &relation.schema),
          tag
        }
      }).collect::<Vec<_>>();

      let collection = DynamicCollection::from_vec(facts, ctx);

      let mut intentional_relation = IntentionalRelation::new();
      intentional_relation.internal_facts = collection;

      relations.insert(name, intentional_relation);
    }
    let mut idb = IntentionalDatabase::new();
    idb.intentional_relations = relations;
    return idb;
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct EDB<Prov: Provenance> {
  pub relations: C_Array<C_Relation<Prov>>
}

impl<Prov: Provenance> std::fmt::Display for EDB<Prov> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.relations)
  }
}

impl<Prov: Provenance> EDB<Prov> {
  pub fn from_extensional_database(edb: &ExtensionalDatabase<Prov>, relation_types: &HashMap<String, TupleType>) -> Self {
    let _guard = flame::start_guard("from_intensional_database_with_filter");
    let iter = edb.extensional_relations.iter().map(|(name, value)| (name.clone(), &value.internal));
    Self { relations: facts_from_iterator(iter, relation_types) }
  }
  pub fn from_extensional_database_with_filter(edb: &ExtensionalDatabase<Prov>, relation_types: &HashMap<String, TupleType>, filter: &Vec<String>) -> Self {
    let _guard = flame::start_guard("from_extensional_database_with_filter");
    let iter = edb.extensional_relations.iter().filter(|(name, _)| filter.contains(name)).map(|(name, value)| (name.clone(), &value.internal));
    Self { relations: facts_from_iterator(iter, relation_types) }
  }
}
