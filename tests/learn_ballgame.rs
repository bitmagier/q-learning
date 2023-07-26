use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x12_5_512_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::log::init_logging;

// Der aktuelle Stand ist eine brauchbare Ausgangslage um das Model und den Lernprozess zu optimieren.
// Aktuell erreichen wir einen Lernerfolg von ca. 66%.
// [2023-07-26T09:40:07Z INFO ] episode 476_869, step count: 3_730_000, epsilon: 0.05, running reward: 3.05
// [2023-07-26T09:40:07Z INFO ] reward distribution: 96x(-1.1..-1.0), 204x(4.9..5.0)
// [2023-07-26T09:40:07Z INFO ] action distribution (last 200_000): 20.2% Nothing, 31.0% North, 11.3% South, 20.0% West, 17.5% East

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()>{
    init_logging();
    
    let mut param = Parameter::default();
    param.max_steps_per_episode = 20;
    param.update_after_actions = 4;
    param.history_buffer_len = 200_000;
    param.epsilon_pure_random_steps = 50_000; 
    param.epsilon_greedy_steps = 2_000_000.0;
    param.episode_reward_history_buffer_len = 300;
    param.epsilon_max = 1.0;
    param.epsilon_min = 0.05;

    const BATCH_SIZE: usize = 512;
    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load(&QL_MODEL_BALLGAME_3x3x12_5_512_PATH);
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