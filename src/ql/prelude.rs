use std::fmt::Display;
use std::rc::Rc;

use plotters::prelude::{CoordTranslate, DrawingBackend};

/// Data type we use to encode an `Action` to feed the model.
/// This one should fit for all usage szenarios (for now).
pub type ModelActionType = u8;

pub trait Action: Display + Sized + Clone + Copy {
    /// ACTION_SPACE = number of possible actions
    const ACTION_SPACE: ModelActionType;
    /// identifying the Action as a unique value in range (0..Self::action_space)
    fn numeric(&self) -> ModelActionType;
    fn try_from_numeric(value: ModelActionType) -> Result<Self, String>;
}

/// Learning environment, modeling the world of a learning agent
pub trait Environment
{
    /// State representation - covering all needs 
    /// Plan:
    /// A. Interface to TensorModel is a Tensor
    /// B. Environment (or State type in environment) implements trait ToTensor
    /// Design goal: Environment shall be suitable for algorithmic generated frames out of a game-engine 
    /// as well as for frames taken originated from a camera
    type S: Clone + DebugVisualizer;
    type A: Action;

    /// Resets the environment to a defined starting point
    fn reset(&mut self);

    /// Current state
    fn state(&self) -> &Self::S;

    /// Performs one time/action-step
    ///
    /// Applies the given `action` to the environment and returns:
    ///   - next state
    ///   - immediate reward earned during performing that step
    ///   - done flag (e.g. game ended)
    ///
    fn step(&mut self, action: Self::A) -> (&Self::S, f32, bool);

    /// Total reward considering the task solved
    /// (expected to be a constant - not a moving target)
    fn total_reward_goal(&self) -> f32;
}


pub const DEFAULT_BATCH_SIZE: usize = 32;

/// 'Physical' AI model abstraction
pub trait QLearningModel<const BATCH_SIZE: usize = DEFAULT_BATCH_SIZE> {
    type E: Environment;

    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [frame_size_x, frame_size_y, world_state_num_frames].
    ///   Representing the current state - this should be the last four frames of the required frame size
    ///   having one pixel encoded in a single float number
    ///
    fn predict_action(&self,
                      state: &<Self::E as Environment>::S,
    ) -> <Self::E as Environment>::A;

    fn batch_predict_future_reward(&self,
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
    fn train(&self,
             state_batch: [&Rc<<Self::E as Environment>::S>; BATCH_SIZE],
             action_batch: [<Self::E as Environment>::A; BATCH_SIZE],
             updated_q_values: [f32; BATCH_SIZE],
    ) -> f32;

    fn write_checkpoint(&self, file: &str) -> String;

    fn read_checkpoint(&self, file: &str);
}
// TODO ^^^ think about decoupling Model from Environment, 
//  so that State-Tensors are passed via type T: ToMultiDimArray<Tensor<f32>> instead of Environment::State.
//  Does it make more sense that way?
//  Advantage: We don't need to pass/handle a full Environment::State object reference from Environment to Model. 
//      Especially the learning algorithm, which need to maintain a state-history buffer, could store less information!
//      It seems logical to reduce the storage requirements of a AI model state object to a minimum. 
//  Disadvantage: We loose the automatic type inference from Environment::State -> Environment::Action when we call Model functions
//  Details:
//      Two options for passing the state to the model seem possible here:
//          A: use <T> T: ToMultiDimArray<Tensor<f32>> instead of Environment::State
//          B: store and use a Tensor<f32> object to pass state, but 
//              Q: is it possible to transform a randomly picked set of stored states to a tensor batch object efficiently?
//              A: No. When creating a Tensor struct in Rust (usually via Tensor::from()), it always copies each single value into the tensor structure one by one.  
//  So what remains to think about is Decoupling Model from Environment and use 
//  - `T: ToMultiDimArray<Tensor<f32>>` for state values
//  - `T: From<u8>, To<u8>` for action values

pub trait DebugVisualizer {
    fn one_line_info(&self) -> String;
    fn plot<DB: DrawingBackend, CT: CoordTranslate>(&self, drawing_area: &mut plotters::drawing::DrawingArea<DB, CT>);
}

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
