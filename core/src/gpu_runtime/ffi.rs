use super::*;
use crate::runtime::provenance::Provenance;

use array::{C_Array, C_String};
use relation::{EDB, IDB};
use std::ffi::c_void;
use stratum::C_Stratum;
use tag::C_Provenance;

extern "C" {
  pub fn execute_stratum_device_raw(
    provenance: *const c_void,
    stratum_set: *const c_void,
    output_relations: *const c_void,
    edbs: *const c_void,
    idbs: *const c_void,
  ) -> *mut c_void;
  pub fn libsclgpu_init();
}

pub unsafe fn execute_stratum_device<Prov: Provenance>(
  provenance: *const C_Provenance,
  stratum_set: *const C_Array<C_Stratum<Prov>>,
  output_relations: *const C_Array<C_String>,
  edbs: *const C_Array<EDB<Prov>>,
  idbs: *const C_Array<IDB<Prov>>,
) -> *const C_Array<IDB<Prov>> {
  let result = execute_stratum_device_raw(
    provenance as *const c_void,
    stratum_set as *const c_void,
    output_relations as *const c_void,
    edbs as *const c_void,
    idbs as *const c_void,
  ) as *const C_Array<IDB<Prov>>;
  if result.is_null() {
    flame::clear();
    panic!("libsclgpu error: execute_stratum_device returned null. See prior logging for exception");
  }
  return result;
}
