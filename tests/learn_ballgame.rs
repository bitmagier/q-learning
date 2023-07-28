use std::fs;
use std::sync::{Arc, RwLock};

use anyhow::Result;

use common::{BATCH_SIZE, CHECKPOINT_FILE_BASE};
use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};
use q_learning_breakout::ql::prelude::QlError;
use q_learning_breakout::util::log::init_logging;

mod common;

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()> {
    use glob::glob;
    
    init_logging();
    
    let mut param = Parameter::default();
    param.max_steps_per_episode = usize::MAX;
    param.update_after_actions = 4;
    param.history_buffer_len = 300_000;
    param.epsilon_pure_random_steps = 100_000; 
    param.epsilon_greedy_steps = 2_000_000.0;
    param.episode_reward_history_buffer_len = 500;
    param.epsilon_max = 1.0;
    param.epsilon_min = 0.07;
    param.lowest_episode_reward_goal_threshold_pct = 0.95;
    
    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);
    let model_instance1 = model_init();
    let model_instance2 = model_init();
    
    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::default()));
    
    for f in glob(&format!("{}*", CHECKPOINT_FILE_BASE.to_str().unwrap())).unwrap() {
        match f {
            Ok(path) => fs::remove_file(path)?,
            Err(_) => ()
        }
    }
    
    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_instance1, model_instance2, CHECKPOINT_FILE_BASE.as_path());
    assert!(!learner.solved());

    let mut episodes_left = 400_000;
    while !learner.solved() {
        learner.learn_episode()?;
        episodes_left -= 1;
        if episodes_left <= 0 {
            return Err(QlError::from("Did not succeed to learn the model till solved"))?
        }
    }

    assert!(learner.solved());
    
    Ok(())
}
