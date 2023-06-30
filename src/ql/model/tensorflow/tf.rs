use std::rc::Rc;
use tensorflow::Tensor;
use crate::ql::prelude::Environment;

pub trait ToTensor {
    /// Diemsions, the object is represented towards the model.
    ///
    /// # Examples
    /// E.g we would use dimensions `[600,600,4]` for an environment state, which is represented 
    /// by a series of four grayscale frames with a frame size of 600x600.  
    fn dims(&self) -> &[u64];

    /// Produce a tensor with the dimensions returned by [Self::dims]  
    fn to_tensor(&self) -> Tensor<f32>;

    /// Produce a tensor of a batch of objects.
    /// The expected dimensionality of the returned tensor is one higher than returned by [Self::to_tensor], having `BATCH_SIZE` as the first axis.  
    fn batch_to_tensor<const BATCH_SIZE: usize>(batch: &[&Rc<Self>; BATCH_SIZE]) -> Tensor<f32>;
}

pub trait TensorflowEnvironment: Environment {
    /// Diemsions, how the state-object is represented towards the model.
    ///
    /// # Examples
    /// E.g we would use dimensions `[600,600,4]` for an environment state, which is represented 
    /// by a series of four grayscale frames with a frame size of 600x600.
    fn state_dims(state: &Self::S) -> &[u64];
    
    /// Produces a tensor from the state.object with dimensions returned by [Self::state_dims]
    fn state_to_tensor(state: &Self::S) -> Tensor<f32>;
    
    /// Produces a tensor of a batch of state-objects.
    /// The expected dimensionality of the returned tensor is one higher than returned by [Self::state_to_tensor], having `BATCH_SIZE` as the first axis.
    fn state_batch_to_tensor<const BATCH_SIZE: usize>(batch: &[&Rc<Self::S>; BATCH_SIZE]) -> Tensor<f32>;
}