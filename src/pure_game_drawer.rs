use image::{ImageBuffer, Rgb};
use crate::breakout::mechanics::BreakoutMechanics;
use crate::ql::breakout_environment::BreakoutDrawer;

pub struct PureGameDrawer {}

impl BreakoutDrawer for PureGameDrawer {
    fn draw(&self, game_state: &BreakoutMechanics) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        todo!()
    }
}
