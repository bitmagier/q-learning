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

// Der aktuelle Stand ist eine brauchbare Ausgangslage um das Model und den Lernprozess zu optimieren.
// Aktuell erreichen wir einen Lernerfolg von ca. 66%.
// [2023-07-26T09:40:07Z INFO ] episode 476_869, step count: 3_730_000, epsilon: 0.05, running reward: 3.05
// [2023-07-26T09:40:07Z INFO ] reward distribution: 96x(-1.1..-1.0), 204x(4.9..5.0)
// [2023-07-26T09:40:07Z INFO ] action distribution (last 200_000): 20.2% Nothing, 31.0% North, 11.3% South, 20.0% West, 17.5% East

// Now. WHY?
// [2023-07-26T15:29:07Z INFO ] episode 437_621, step count: 3_290_000, epsilon: 0.10, running reward: 2.62
// [2023-07-26T15:29:07Z INFO ] reward distribution: 117x(-1.1..-1.0), 183x(4.9..5.0)
// [2023-07-26T15:29:07Z INFO ] action distribution (last 100_000): Nothing 13.5%, North 31.5%, South 9.0%, West 23.6%, East 22.4%
//
// [2023-07-26T18:35:59Z INFO ] episode 4_327_487, step count: 30_710_000, epsilon: 0.10, running reward: 3.06
// [2023-07-26T18:35:59Z INFO ] reward distribution: 95x(-1.1..-1.0), 205x(4.9..5.0)
// [2023-07-26T18:35:59Z INFO ] action distribution (last 100_000): Nothing 15.2%, North 32.1%, South 8.0%, West 22.2%, East 22.5%
//
// Ist das Model überhaupt lernbar?
// Seems so, but we stuck in local minima corners where a NON-action is slightly better than anything else
// So: 
// const MAX_STEPS: usize = 20;
// param.epsilon_greedy_steps: 1_00_000.0 => 1_500_000.0
// Rewards:
// if let MoveResult::Legal { done: true } = r {
//             (self.state(), 5.0, true)
//         } else if self.state.steps >= MAX_STEPS {
//             (self.state(), -5.0, true) // was -1.0
//         } else if let MoveResult::Legal { done: false } = r {
//             (self.state(), -0.01, false)
//         } else if let MoveResult::Illegal = r {
//             (self.state(), -0.5, false)

// next: remove history frames, increase history buffer to 200_000, increase greedy_frames to 2M
#[test]
fn test_learn_ballgame_until_mastered() -> Result<()> {
    use glob::glob;
    
    init_logging();
    
    let mut param = Parameter::default();
    param.max_steps_per_episode = usize::MAX;
    param.update_after_actions = 4;
    param.history_buffer_len = 200_000;
    param.epsilon_pure_random_steps = 100_000; 
    param.epsilon_greedy_steps = 2_000_000.0;
    param.episode_reward_history_buffer_len = 500;
    param.epsilon_max = 1.0;
    param.epsilon_min = 0.10;
    
    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);
    let model_instance1 = model_init();
    let model_instance2 = model_init();
    
    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::new()));
    
    for f in glob(&format!("{}*", CHECKPOINT_FILE_BASE.to_str().unwrap())).unwrap() {
        match f {
            Ok(path) => fs::remove_file(path)?,
            Err(_) => ()
        }
    }
    
    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_instance1, model_instance2, CHECKPOINT_FILE_BASE.as_path());
    assert!(!learner.solved());

    let mut episodes_left = 500_000;
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
