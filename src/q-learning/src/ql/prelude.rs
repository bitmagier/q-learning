use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::rc::Rc;

use anyhow::Result;
use console_engine::screen::Screen;

/// Data type we use to encode an `Action` to feed the model.
/// This one should fit for all usage szenarios (for now).
pub type ModelActionType = u8;

pub trait Action: Display + Sized + Clone + Copy + Hash + PartialEq + Eq {
    /// Number of possible actions
    const ACTION_SPACE: ModelActionType;
    /// Identifying the Action as a unique value in range (0..Self::action_space)
    fn numeric(&self) -> ModelActionType;
    fn try_from_numeric(value: ModelActionType) -> Result<Self>;
}

/// Learning environment, modeling the world of a learning agent
pub trait Environment {
    /// State representation - covering all needs
    ///
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



pub trait DebugVisualizer {
    fn one_line_info(&self) -> String;
    fn render_to_console(&self) -> Screen;
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
