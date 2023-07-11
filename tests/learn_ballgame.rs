use std::sync::{Arc, RwLock};

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_5x5x3_4_32_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::init_logging;

#[test]
fn test_learn_ballgame_until_mastered() {
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = 30;
    // param.epsilon_greedy_frames = 200_000.0; 

    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment>::load(&QL_MODEL_BALLGAME_5x5x3_4_32_PATH);
    let model_instance1 = model_init();
    let model_instance2 = model_init();
    let checkpoint_file = tempfile::tempdir().unwrap().into_path().join("test_learner_ckpt");
    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::new()));
    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_instance1, model_instance2, &checkpoint_file);
    assert!(!learner.solved());

    while !learner.solved() {
        learner.learn_episode();
    }

    assert!(learner.solved());
}