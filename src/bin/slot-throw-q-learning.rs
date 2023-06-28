use console_engine::screen::Screen;
use q_learning_breakout::environment::slot_throw_environment::SlotThrowEnvironment;
use q_learning_breakout::ql::model::q_learning_model_600x600x4to3::QLearningModel600x600x4to3;
use q_learning_breakout::util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    util::init_logging();

    let mut screen = Screen::new(30,30);
    let mut environment = SlotThrowEnvironment::new();

    let model = QLearningModel600x600x4to3::<SlotThrowEnvironment>::init();



    Ok(())
}
