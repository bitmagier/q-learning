use std::fmt::Display;
use std::rc::Rc;

use tensorflow::Tensor;

pub trait State {
    fn to_tensor(&self) -> Tensor<f32>;
    fn batch_to_tensor<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32>;
}

pub type ModelActionType = u8;

/// Action type placeholder
pub trait Action: Display + Sized + Clone + Copy {
    /// ACTION_SPACE = number of possible actions
    const ACTION_SPACE: ModelActionType;
    /// identifying the Action as a unique value in range (0..Self::action_space)
    fn numeric(&self) -> ModelActionType;
    fn try_from_numeric(value: ModelActionType) -> Result<Self, String>;
}

pub trait Environment {
    type State: State;
    type Action: Action;

    /// Resets the environment to a defined starting point
    fn reset(&mut self);

    /// The Action-variant which represents no particular action
    fn no_action() -> Self::Action;

    /// Performs one time/action-step
    ///
    /// Applies the given ` action` to the environment and returns:# Arguments
    ///   - next state
    ///   - immediate reward earned during performing that step
    ///   - done flag (e.g. game ended)
    ///
    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, f32, bool);

    /// Total reward considering the task solved
    /// (expected to be a constant - not a moving target)
    fn total_reward_goal() -> f32;
}
