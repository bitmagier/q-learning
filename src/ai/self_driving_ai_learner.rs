
// TODO needs an independent presentation layer (decoupled from the egui event loop) for drawing the game state

use crate::breakout::mechanics::BreakoutMechanics;


type AiResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Directly connected to GameMechanics and drives the speed of the game with it's response
pub struct SelfDrivingAiLearner {
    mechanics: BreakoutMechanics
}

impl SelfDrivingAiLearner {
    pub fn run(&mut self) -> AiResult<()>{
        return Ok(())
    }
}
