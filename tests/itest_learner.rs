use std::sync::{Arc, atomic, RwLock};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use q_learning_breakout::ql::ballgame_test_environment::BallGameTestEnvironment;
use q_learning_breakout::ql::learn::self_driving_q_learner::{Parameter, SelfDrivingQLearner};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x3_4_32_PATH, QLearningTensorflowModel};
use q_learning_breakout::util::init_logging;

#[test]
// TODO investigate why there is no learning success => running_reward even gets lower (below zero) over time
// make sure logging is switched off
fn test_learn_until_mastered() {
    init_logging();
    
    let mut param = Parameter::default();
    param.max_steps_per_episode = 14;
    
    let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment>::load(&QL_MODEL_BALLGAME_3x3x3_4_32_PATH);
    let model_instance1 = model_init();
    let model_instance2 = model_init();
    let checkpoint_file = tempfile::tempdir().unwrap().into_path().join("test_learner_ckpt");
    let environment = Arc::new(RwLock::new(BallGameTestEnvironment::new()));
    let mut learner = SelfDrivingQLearner::new(Arc::clone(&environment), param, model_instance1, model_instance2, &checkpoint_file);
    assert!(!learner.solved());

    let finish_signal = Arc::new(AtomicBool::new(false));

    // let console_finish_signal = Arc::clone(&finish_signal);
    // let console_thread_handle = thread::spawn(move || {
    //     let mut engine = ConsoleEngine::init(3, 3, 30).unwrap();
    //     loop {
    //         engine.wait_frame(); // wait for next frame + capture inputs
    //         let debug_screen = environment.read().unwrap().state().get_debug_screen();
    //         engine.set_screen(&debug_screen);
    //         
    //         engine.draw();
    //         
    //         if engine.is_key_held(KeyCode::Char(' ')) {
    //             thread::sleep(Duration::from_secs(1));
    //         }
    //         if engine.is_key_pressed(KeyCode::Esc) {
    //             console_finish_signal.store(true, atomic::Ordering::Relaxed);
    //             break;
    //         }
    //     }
    // });

    while !learner.solved() && !finish_signal.load(Relaxed) {
        learner.learn_episode();
    }
    finish_signal.store(true, Relaxed);

    // console_thread_handle.join();

    assert!(learner.solved());
}