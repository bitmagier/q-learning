use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_5x5x3_4_256_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::init_logging;

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()>{
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = 28;
    param.update_after_actions = 4;
    param.history_buffer_len = 200_000;
    param.epsilon_greedy_steps = 1_000_000.0;
    param.episode_reward_history_buffer_len = 500;
    param.update_target_network_after_num_steps = 50_000;

    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, 256>::load(&QL_MODEL_BALLGAME_5x5x3_4_256_PATH);
    let model_instance1 = model_init();
    let model_instance2 = model_init();
    let checkpoint_file = tempfile::tempdir().unwrap().into_path().join("test_learner_ckpt");
    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::new()));
    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_instance1, model_instance2, &checkpoint_file);
    assert!(!learner.solved());

    while !learner.solved() {
        learner.learn_episode()?;
    }

    assert!(learner.solved());
    
    Ok(())
}