pub trait ExternalTensor {

}

pub struct DynamicExternalTensor(Box<dyn ExternalTensor>);

pub struct TensorSymbol {

}

pub trait TensorRegistry {

}

pub struct DynamicTensorRegistry(Box<dyn TensorRegistry>);
