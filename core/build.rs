extern crate cc;
extern crate cmake;
extern crate lalrpop;

use std::env;

fn main() {
  // lalrpop::Configuration::new()
  //   .generate_in_source_tree()
  //   .process()
  //   .unwrap();

  //let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  //println!("cargo:rerun-if-changed=src/gpu_runtime/mod.rs");
  //cbindgen::Builder::new()
  //  .with_crate(crate_dir)
  //  .generate()
  //  .expect("Unable to generate bindings")
  //  .write_to_file("src/gpu_runtime/libsclgpu/new_bindings.h");

  let profile = env::var("PROFILE").unwrap();
  if profile == "release" {
    println!("cargo:rustc-link-search=native=/home/paulbib/scallop-v2/core/src/gpu_runtime/libsclgpu/cmake-release");
  } else {
    println!("cargo:rustc-link-search=native=/home/paulbib/scallop-v2/core/src/gpu_runtime/libsclgpu/cmake-debug");
  }
  println!("cargo:rustc-link-lib=static=sclgpu");

  println!("cargo:rerun-if-changed=src/gpu_runtime/libsclgpu/cmake-debug/libsclgpu.a");
  println!("cargo:rerun-if-changed=src/gpu_runtime/libsclgpu/cmake-release/libsclgpu.a");

  println!("cargo:rustc-link-lib=stdc++");
  println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
  println!("cargo:rustc-link-lib=cudart_static");
  println!("cargo:rustc-link-lib=cudadevrt");
}
