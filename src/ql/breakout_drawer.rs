use image::{ImageBuffer, Rgb};
use crate::breakout::mechanics::BreakoutMechanics;


pub trait BreakoutDrawer {
    fn draw(&self, game_state: &BreakoutMechanics) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
}

// TODO needs an independent presentation layer (decoupled from the egui event loop) for drawing the game state
// TODO render attempts if wish expressed (keypress 'r' maybe)
