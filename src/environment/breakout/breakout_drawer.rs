#![allow(unused)]

use image::{ImageBuffer, Rgb};

use crate::environment::breakout::mechanics::BreakoutMechanics;

pub struct BreakoutDrawer {
    frame_size_x: usize,
    frame_size_y: usize,
}

impl BreakoutDrawer {
    pub fn new(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
        }
    }
    pub fn draw(&self, _game_state: &BreakoutMechanics) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        // TODO use plotter
        unimplemented!()
    }
}


// TODO needs an independent presentation layer (decoupled from the egui event loop) for drawing the game state
// TODO render attempts if wish expressed (keypress 'r' maybe)
