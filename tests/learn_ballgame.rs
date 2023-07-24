use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_32_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::log::init_logging;


// [2023-07-24T16:17:16Z INFO ] episode 50_379, step count: 920_000, epsilon: 0.17, running reward: 4.66
// [2023-07-24T16:17:16Z INFO ] reward distribution: 8x(-2.2..-2.2), 10x(-1.7..-1.7), 10x(-1.3..-1.3), 12x(-0.8..-0.8), 51x(9.2..10.0), 9x(noise)
// [2023-07-24T16:17:16Z INFO ] action distribution (last 500_000): (52.0%Nothing) (16.7%North) (9.4%South) (10.0%West) (11.9%East)

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()>{
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = 28;
    // param.update_after_actions = 1;
    param.history_buffer_len = 500_000;
    param.epsilon_greedy_steps = 1_000_000.0;
    param.episode_reward_history_buffer_len = 100;

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