use std::rc::Rc;

use anyhow::Result;

use crate::ql::prelude::Environment;

pub const DEFAULT_BATCH_SIZE: usize = 32;


/// Generic capability to produce a multi dimensional array out of an object or a batch of objects.
///
/// Type parameter `D` stands for the produced multi-dimensional array type (e.g. `Tensor<f32>`)
pub trait ToMultiDimArray<D> {
    /// Dimensions of the produced array (for a single object).
    ///
    /// # Examples
    /// E.g we would use dimensions `[600,600,4]` for an environment state, which is represented
    /// by a series of four grayscale frames with a frame size of 600x600.
    fn dims(&self) -> &[u64];

    /// Produces a multi dimensional array of the associated type `T` with the dimensions returned by [Self::dims]
    fn to_multi_dim_array(&self) -> D;
    /// Produces a multi dimensional array from a batch of objects.
    /// The expected dimensionality of the returned tensor usually has one axis (with len = `BATCH_SIZE`)
    /// more than for a single object (as returned by [Self::to_tensor]).
    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Rc<Self>; N]) -> D;
}

/// 'Physical' AI model abstraction
pub trait DeepQLearningModel<const BATCH_SIZE: usize = DEFAULT_BATCH_SIZE> {
    type E: Environment;

    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [frame_size_x, frame_size_y, world_state_num_frames].
    ///   Representing the current state - this should be the last four frames of the required frame size
    ///   having one pixel encoded in a single float number
    ///
    fn predict_action(
        &self,
        state: &<Self::E as Environment>::S,
    ) -> <Self::E as Environment>::A;

    fn batch_predict_max_future_reward(
        &self,
        states: [&Rc<<Self::E as Environment>::S>; BATCH_SIZE],
    ) -> [f32; BATCH_SIZE];

    /// Performs a single training step using a a batch of data.
    /// Returns the model's loss
    ///
    /// # Arguments
    /// * `state_batch` Tensor [BATCH_SIZE, frame_size_x, frame_size_y, world_state_num_frames]
    /// * `action_batch` Tensor [BATCH_SIZE, 1]
    /// * `updated_q_values` Tensor [BATCH_SIZE, 1]
    ///
    /// # Returns
    ///   calculated loss
    ///
    fn train(
        &self,
        state_batch: [&Rc<<Self::E as Environment>::S>; BATCH_SIZE],
        action_batch: [<Self::E as Environment>::A; BATCH_SIZE],
        updated_q_values: [f32; BATCH_SIZE],
    ) -> Result<()>;

    fn write_checkpoint(
        &self,
        file: &str,
    ) -> Result<String>;

    /// That function is currently more like a wish than a doable thing - at least with Tensorflow
    fn read_checkpoint(
        &self,
        file: &str,
    ) -> Result<()>;
}

