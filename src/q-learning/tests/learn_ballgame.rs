use std::fs;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use common::{BATCH_SIZE, CHECKPOINT_FILE_BASE};
use q_learning_breakout::environment::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_tf_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow_python::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};
use q_learning_breakout::ql::prelude::QlError;
use q_learning_breakout::util::log::init_logging;

mod common;

// We are close, but not done yet:
// [2023-07-29T11:08:00Z INFO ]
//    episode: 211_057, steps: 1_920_000, 𝛾=0.90, 𝜀=0.18, reward_goal: {mean >= 9.5, low >= 9.0}, current_rewards: {mean: 9.6, low: -14.2}
//    reward_distribution: 24x(7.9..8.0), 100x(8.8..9.0), 369x(9.8..10.0), 7x(noise)
//    action_distribution (of last 200_000): o 8.6%, ↑ 44.0%, ↓ 5.0%, ← 21.2%, → 21.2%
// ...
// [2023-07-29T11:43:44Z INFO ]
//    episode: 1_356_318, steps: 6_840_000, 𝛾=0.90, 𝜀=0.15, reward_goal: {mean >= 9.5, low >= 9.0}, current_rewards: {mean: 9.6, low: -13.2}
//    reward_distribution: 14x(7.9..8.0), 89x(8.8..9.0), 392x(9.7..10.0), 5x(noise)
//    action_distribution (of last 200_000): o 4.3%, ↑ 49.3%, ↓ 3.4%, ← 21.4%, → 21.6%

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()> {
    use glob::glob;

    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = usize::MAX;
    param.gamma = 0.95;
    //  TODO 2_000
    param.update_target_network_after_num_steps = 20_000;
    param.update_after_actions = 4;
    param.history_buffer_len = 200_000;
    param.epsilon_pure_random_steps = 100_000;
    param.epsilon_greedy_steps = 2_500_000.0;
    param.episode_reward_history_buffer_len = 500;
    param.epsilon_max = 1.0;
    param.epsilon_min = 0.15;
    param.lowest_episode_reward_goal_threshold_pct = 0.75;

    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load_model(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);
    let model_instance1 = model_init()?;
    let model_instance2 = model_init()?;

    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::default()));

    for f in glob(&format!("{}*", CHECKPOINT_FILE_BASE.to_str().unwrap())).unwrap() {
        match f {
            Ok(path) => fs::remove_file(path)?,
            Err(_) => (),
        }
    }

    let mut learner = SelfDrivingQLearner::new(
        Arc::clone(&environment),
        param,
        model_instance1,
        model_instance2,
        CHECKPOINT_FILE_BASE.clone(),
    );
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
