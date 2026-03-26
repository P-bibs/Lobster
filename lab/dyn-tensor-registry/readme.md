# Dynamic Tensor Registry

## Specification

- Tensors of any type can be imported into Scallop
- Scallop only provides a trait specification, no actual type is implemented in Scallop core
  - i.e. there is no dependency on crates such as `tch-rs`
- Dynamic evaluation can be performed during execution, without access to their original type
- Evaluation needs to be carried through original sources, as additional data structures are needed
