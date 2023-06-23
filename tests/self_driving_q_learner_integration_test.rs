use std::rc::Rc;
use tempfile::NamedTempFile;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::prelude::Environment;

// TODO Simple simulated Test Environment

struct TestEnvironment {}
impl Environment for TestEnvironment {
    type State = ();
    type Action = ();

    fn reset(&mut self) {
        todo!()
    }

    fn no_action() -> Self::Action {
        todo!()
    }

    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, f32, bool) {
        todo!()
    }

    fn total_reward_goal() -> f32 {
        todo!()
    }
}

#[test]
fn itest_self_driving_q_learner() {
    let checkpoint_file = NamedTempFile::new()?;
    let learner = SelfDrivingQLearner::from_scratch(
        TestEnvironment::new(),
        Parameter::default(),
        checkpoint_file.path());
    learner.learn_until_mastered();
}
