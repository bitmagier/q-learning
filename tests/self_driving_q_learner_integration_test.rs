use tempfile::NamedTempFile;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::util;

mod slot_throw_environment;

#[test]
fn itest_self_driving_q_learner() {
    util::init_logging();
    let checkpoint_file = NamedTempFile::new().unwrap();
    let mut learner = SelfDrivingQLearner::from_scratch(
        slot_throw_environment::SlotThrowEnvironment::new(),
        Parameter::default(),
        checkpoint_file.path());

    learner.learn_until_mastered();
}
