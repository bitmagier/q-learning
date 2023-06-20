use std::rc::Rc;

pub trait Environment {
    type State;
    type Action;
    type Reward;

    fn reset(&mut self);

    fn no_action(&self) -> Self::Action;

    /// Performs one time/action-step
    ///
    /// Applies the given ` action` to the environment and returns:# Arguments
    ///   - next state
    ///   - immediate reward earned during performing that step
    ///   - done flag (e.g. game ended)
    ///
    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, Self::Reward, bool);
}

