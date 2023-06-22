use image::{ImageBuffer, Rgb};
use crate::breakout::mechanics::BreakoutMechanics;

pub trait BreakoutDrawer {
    fn draw(&self, game_state: &BreakoutMechanics) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
}
