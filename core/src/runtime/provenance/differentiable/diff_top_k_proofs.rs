use std::convert::TryFrom;
use crate::common::foreign_tensor::*;
use crate::gpu_runtime::array::C_Array;
use crate::gpu_runtime::tag::{C_DiffTopKProofs, C_Proof, C_Provenance};
use crate::utils::*;

use super::*;

pub struct DiffTopKProofsProvenance<T: FromTensor, P: PointerFamily> {
  pub k: usize,
  pub storage: DiffProbStorage<T, P>,
  pub disjunctions: P::Cell<Disjunctions>,
  pub wmc_with_disjunctions: bool,
}

impl<T: FromTensor, P: PointerFamily> Clone for DiffTopKProofsProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      storage: self.storage.clone_internal(),
      disjunctions: P::clone_cell(&self.disjunctions),
      wmc_with_disjunctions: self.wmc_with_disjunctions,
    }
  }
}

impl<T: FromTensor, P: PointerFamily> DiffTopKProofsProvenance<T, P> {
  pub fn new(k: usize, wmc_with_disjunctions: bool) -> Self {
    Self {
      k,
      storage: DiffProbStorage::new(),
      disjunctions: P::new_cell(Disjunctions::new()),
      wmc_with_disjunctions,
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.storage.input_tags()
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<T: FromTensor, P: PointerFamily> DNFContextTrait for DiffTopKProofsProvenance<T, P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.storage.fact_probability(id)
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    P::get_cell(&self.disjunctions, |d| d.has_conflict(pos_facts))
  }
}

impl<T: FromTensor, P: PointerFamily> Provenance for DiffTopKProofsProvenance<T, P> {
  fn print_storage(&self) {
    self.storage.print_storage();
  }
  type Tag = DNFFormula;

  type InputTag = InputExclusiveDiffProb<T>;

  type OutputTag = OutputDiffProb;

  type FFITag = C_Proof;
  fn to_ffi_tag(tag: &Self::Tag) -> Self::FFITag {
    let lit_to_int = |l: &Literal| match l {
      Literal::Pos(id) => i32::try_from(*id).unwrap(),
      Literal::Neg(id) => -i32::try_from(*id).unwrap()
    };
    match tag.clauses.get(0) {
      Some(proof) => C_Proof {
        literals: C_Array::new(proof.literals.iter().map(lit_to_int).collect()),
        empty: false,
      },
      None => C_Proof {
        literals: C_Array::empty(),
        empty: true,
      },
    }
  }
  fn from_ffi_tag(tag: &Self::FFITag) -> Self::Tag {
    let int_to_lit = |n: &i32| {
      if *n < 0 {
        Literal::Neg((*n * -1) as usize)
      } else {
        Literal::Pos(*n as usize)
      }
    };
    match tag.empty {
      false => DNFFormula::new(vec![Clause::new(tag.literals.as_slice().iter().map(int_to_lit).collect())]),
      true => DNFFormula::new(vec![]),
    }
  }
  fn to_ffi_provenance(ctxes: &Vec<Self>) -> C_Provenance {
    C_Provenance::DiffTopKProofs(C_DiffTopKProofs::from(ctxes))
  }

  fn name(&self) -> String {
    format!("diff-top-k-proofs")
  }

  fn tagging_fn(&self, input_tag: Self::InputTag) -> Self::Tag {
    let InputExclusiveDiffProb {
      prob,
      external_tag,
      exclusion,
    } = input_tag;

    // First store the probability and generate the id
    let fact_id = self.storage.add_prob(prob, external_tag);

    // Store the mutual exclusivity
    if let Some(disjunction_id) = exclusion {
      P::get_cell_mut(&self.disjunctions, |d| d.add_disjunction(disjunction_id, fact_id));
    }

    // Finally return the formula
    DNFFormula::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    // Get the number of variables that requires grad
    let num_var_requires_grad = self.storage.num_input_tags();
    let s = DualNumberSemiring::new(num_var_requires_grad);
    let v = |i: &usize| {
      let (real, external_tag) = self.storage.get_diff_prob(i);

      // Check if this variable `i` requires grad or not
      if external_tag.is_some() {
        s.singleton(real.clone(), i.clone())
      } else {
        s.constant(real.clone())
      }
    };
    let wmc_result = if self.wmc_with_disjunctions {
      P::get_cell(&self.disjunctions, |disj| t.wmc_with_disjunctions(&s, &v, disj))
    } else {
      t.wmc(&s, &v)
    };
    let prob = wmc_result.real;
    let deriv = wmc_result
      .deriv
      .iter()
      .map(|(id, weight)| (id, *weight))
      .collect::<Vec<_>>();
    OutputDiffProb(prob, deriv)
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero()
  }

  fn one(&self) -> Self::Tag {
    self.base_one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_add(t1, t2, self.k)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_mult(t1, t2, self.k)
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.top_k_negate(t, self.k))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let v = |i: &usize| self.storage.get_prob(i);
    t.wmc(&RealSemiring::new(), &v)
  }
}
