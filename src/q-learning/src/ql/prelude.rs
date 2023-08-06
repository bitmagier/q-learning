use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::rc::Rc;

use anyhow::Result;
use console_engine::screen::Screen;

/// Data type we use to encode an `Action` to feed the model.
/// This one should fit for all usage szenarios (for now).
pub type ModelActionType = u8;

pub trait Action: Display + Sized + Clone + Copy + Hash + PartialEq + Eq {
    /// ACTION_SPACE = number of possible actions
    const ACTION_SPACE: ModelActionType;
    /// identifying the Action as a unique value in range (0..Self::action_space)
    fn numeric(&self) -> ModelActionType;
    fn try_from_numeric(value: ModelActionType) -> Result<Self>;
}

/// Learning environment, modeling the world of a learning agent
pub trait Environment {
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

    /// Convenience wrapper around [Self::state]
    fn state_as_rc(&self) -> Rc<Self::S> { Rc::new(self.state().clone()) }

    /// Performs one time/action-step.    
    ///
    /// Applies the given `action` to the environment and returns:
    ///   - next state
    ///   - immediate reward earned during performing that step
    ///   - done flag (e.g. game ended)
    ///
    fn step(
        &mut self,
        action: Self::A,
    ) -> (&Self::S, f32, bool);

    /// Convenience wrapper around [Self::step] returning an [Rc] with a copy of the state.
    /// This should match the typical use-case.
    fn step_as_rc(
        &mut self,
        action: Self::A,
    ) -> (Rc<Self::S>, f32, bool) {
        let (state, reward, done) = self.step(action);
        (Rc::new(state.clone()), reward, done)
    }

    /// Average reward to reach over all episodes
    /// (expected to be a constant - not a moving target)
    fn episode_reward_goal_mean(&self) -> f32;
}

pub const DEFAULT_BATCH_SIZE: usize = 32;

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

pub trait DebugVisualizer {
    fn one_line_info(&self) -> String;
    fn render_to_console(&self) -> Screen;
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

#[derive(Debug)]
pub struct QlError(pub String);

impl QlError {
    pub fn from(msg: &str) -> Self { QlError(msg.to_string()) }
}

impl Display for QlError {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for QlError {}