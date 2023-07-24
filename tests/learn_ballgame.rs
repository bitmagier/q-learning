use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_32_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::log::init_logging;

// -----------------------------------------------------------------------------------------------
//         self.add(layers.Flatten(name='flatten'))
//         self.add(layers.Dense(256, activation='sigmoid', name='full_layer1'))
//         self.add(layers.Dense(256, activation='sigmoid', name='full_layer2'))
//         self.add(layers.Dense(256, activation='sigmoid', name='full_layer3'))
//         self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))
//
// episode 120_174, step count: 3_010_000, epsilon: 0.10, running reward: 0.18
// reward distribution: 5x(-2.2..-2.2), 6x(-1.7..-1.7), 25x(-1.3..-1.3), 28x(-0.8..-0.8), 23x(-0.3..-0.3), 7x(9.8..10.0), 6x(noise)
// action distribution (last 200000): (43.4%Nothing) (7.1%North) (5.9%South) (21.4%West) (22.2%East)
// -----------------------------------------------------------------------------------------------

#[test]
fn test_learn_ballgame_until_mastered() -> Result<()>{
    init_logging();

    let mut param = Parameter::default();
    param.max_steps_per_episode = 28;
    param.update_after_actions = 1;
    param.history_buffer_len = 200_000;
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