use std::sync::{Arc, RwLock};

use anyhow::Result;

use common::consts::BATCH_SIZE;
use ql::prelude::QlError;
use ql::test::ballgame_test_environment::BallGameTestEnvironment;
use ql_in_rust::learn::self_driving_tf_q_learner::{Parameter, SelfDrivingQLearner};
use ql_in_rust::ml_model::tensorflow_python::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};
use ql_in_rust::log::init_logging;
use crate::common::consts::CHECKPOINT_FILE_BASE;

mod common;

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()> {
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = usize::MAX;
    param.gamma = 0.95;
    param.update_target_network_after_num_steps = 5_000;
    param.update_after_actions = 4;
    param.history_buffer_len = 200_000;
    param.epsilon_pure_random_steps = 100_000;
    param.epsilon_greedy_steps = 2_500_000.0;
    param.episode_reward_history_buffer_len = 500;
    param.epsilon_max = 1.0;
    param.epsilon_min = 0.15;
    param.lowest_episode_reward_goal_threshold_pct = 0.75;

    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load_model(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);

    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::default()));

    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_init, CHECKPOINT_FILE_BASE.clone())?;
    assert!(!learner.solved());

    let mut episodes_left = 1_500_000;
    while !learner.solved() {
        learner.learn_episode()?;
        episodes_left -= 1;
        if episodes_left <= 0 {
            return Err(QlError::from("Did not succeed to learn the model till solved"))?;
        }
    }

    assert!(learner.solved());

    Ok(())
}
