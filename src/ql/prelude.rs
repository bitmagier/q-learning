use std::fmt::Display;
use std::rc::Rc;

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
    type S;
    type A: Action;

    /// Resets the environment to a defined starting point
    fn reset(&mut self);

    /// Current state
    fn state(&self) -> Rc<Self::S>;
    
    /// Performs one time/action-step
    ///
    /// Applies the given ` action` to the environment and returns:# Arguments
    ///   - next state
    ///   - immediate reward earned during performing that step
    ///   - done flag (e.g. game ended)
    ///
    fn step(&mut self, action: Self::A) -> (Rc<Self::S>, f32, bool);

    /// Total reward considering the task solved
    /// (expected to be a constant - not a moving target)
    fn total_reward_goal(&self) -> f32;
}

/// 'physical' AI model abstraction
pub trait QLearningModel<const BATCH_SIZE: usize> {
    type E: Environment;
    
    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [frame_size_x, frame_size_y, world_state_num_frames].
    ///   Representing the current state - this should be the last four frames of the required frame size
    ///   having one pixel encoded in a single float number
    ///
    fn predict_action(&self,
                      state: &Rc<<Self::E as Environment>::S>,
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