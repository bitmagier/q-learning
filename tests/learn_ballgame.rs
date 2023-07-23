use std::sync::{Arc, RwLock};

use anyhow::Result;

use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_5x5x3_4_256_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::log::init_logging;


// def __init__(self, *args, **kwargs):
//         super(QLearningModel_BallGame_5x5x3_4_256, self).__init__(*args, **kwargs)
//         # Ideas:
//         # - decrease learning rate while learning => tf.keras.optimizers.schedules.LearningRateSchedule
//         self.add(tf.keras.Input(shape=(INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS,)))
//         self.add(layers.Conv2D(64, 3, strides=1, activation='relu', name='convolution_layer1'))
//         self.add(layers.Conv2D(32, 1, strides=1, activation='relu', name='convolution_layer2'))
//         self.add(layers.Flatten(name='flatten'))
//         self.add(layers.Dense(256, activation='relu', name='full_layer1'))
//         self.add(layers.Dense(256, activation='softmax', name='full_layer2'))
//         self.add(layers.Dense(256, activation='softmax', name='full_layer3'))
//         self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))
// 
//         self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
//                      # Using huber loss for stability
//                      loss=keras.losses.Huber(),
//                      metrics=['accuracy'],
//                      )
//
// Its not working as intended
// episode 407_851, step count (frames): 9_350_000, epsilon: 0.10, running reward: -0.47
// reward distribution: 16x(-4.4..-4.4), 53x(-4.0..-4.0), 97x(-3.6..-3.6), 107x(-3.2..-3.2), 62x(-2.8..-2.8), 32x(7.3..8.4), 92x(8.5..9.9), 41x(noise)
//
// Mit nur einem 1x1 conv2d Layer sieht es auch nicht besser aus:
// episode 53_059, step count (frames): 1_150_000, epsilon: 0.10, running reward: -1.23
// reward distribution: 19x(-4.4..-4.4), 41x(-4.0..-4.0), 71x(-3.6..-3.6), 94x(-3.2..-3.2), 56x(-2.8..-2.8), 20x(8.0..8.8), 73x(8.9..9.9), 126x(noise)


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