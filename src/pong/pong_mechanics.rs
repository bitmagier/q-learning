use std::rc::Rc;
use crate::pong::game_api::{GameState, GameInput, Pong};

pub struct PongMechanics {}

impl PongMechanics {
    pub fn new() -> PongMechanics {
        todo!()
    }
}

impl Pong for PongMechanics {
    fn state(&self) -> Rc<GameState> {
        todo!()
    }

    fn time_step(&mut self, input: GameInput) {
        todo!()
    }
}
