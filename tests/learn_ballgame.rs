use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_32_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::log::init_logging;


// we are coming closer
// [2023-07-24T18:45:25Z INFO ] episode 128_446, step count: 1_890_000, epsilon: 0.20, running reward: 6.73
// [2023-07-24T18:45:25Z INFO ] reward distribution: 15x(-1.9..-1.2), 5x(-0.9..-0.8), 22x(8.9..9.5), 48x(9.9..10.0), 10x(noise)
// [2023-07-24T18:45:25Z INFO ] action distribution (last 500_000): (6.3% Nothing) (35.3% North) (22.6% South) (16.7% West) (19.1% East)

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()>{
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = 24;
    param.update_after_actions = 4;
    param.history_buffer_len = 500_000;
    param.epsilon_greedy_steps = 2_000_000.0;
    param.episode_reward_history_buffer_len = 100;
    param.epsilon_min = 0.15;

    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, 32>::load(&QL_MODEL_BALLGAME_3x3x4_5_32_PATH);
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