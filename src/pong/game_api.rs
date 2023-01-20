#![allow(unused)]

use std::rc::Rc;
use std::time::Duration;
use egui::Vec2;
use crate::pong::pong_mechanics::PongMechanics;


pub const MODEL_LEN_X: f32 = 600.0;
pub const MODEL_LEN_Y: f32 = 800.0;


#[derive(Copy, Clone)]
pub struct Vector2d {
    pub x: f32,
    pub y: f32
}
impl Vector2d {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}
impl From<(f32, f32)> for Vector2d {
    fn from(value: (f32, f32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}
impl From<(isize, isize)> for Vector2d {
    fn from(value: (isize, isize)) -> Self {
        Self {
            x: value.0 as f32,
            y: value.1 as f32
        }
    }
}

pub type Coordinate = Vector2d;

pub trait Pong {
    fn time_step(&mut self, input: GameInput) -> GameState;
}

#[derive(Clone)]
pub struct GameState {
    // x = 0 = left side; y = 0 = bottom
    pub bricks: Vec<Brick>,
    pub ball: Ball,
    pub panel: Panel,
    pub finished: bool,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            bricks: PongMechanics::initial_bricks(),
            ball: PongMechanics::initial_ball(),
            panel: PongMechanics::initial_panel(),
            finished: false,
        }
    }
}


#[derive(Copy, Clone)]
pub struct GameInput {
    pub control: PanelControl,
}

impl GameInput {
    pub fn new() -> Self {
        Self { control: PanelControl::None }
    }
}

#[derive(Copy, Clone)]
pub enum PanelControl {
    None,
    AccelerateLeft,
    AccelerateRight,
    Exit,
}

#[derive(Clone)]
pub struct Brick {
    // 0,0 is in the lower left corner
    pub lower_left: Coordinate,
    pub upper_right: Coordinate,
}

/// A ball is a perfect round 2D structure
#[derive(Clone)]
pub struct Ball {
    pub center: Coordinate,
    pub radius: f32,
    pub direction: Vector2d,
    pub speed: f32,
}

#[derive(Clone)]
pub struct Panel {
    pub center_pos_x: f32,
    pub center_pos_y: f32,
    pub size_x: f32,
    pub size_y: f32,
    pub speed: f32,
}


pub trait Assert {
    fn assert(&self);
}
impl Assert for Panel {
    fn assert(&self) {
        assert!(self.center_pos_x - self.size_x / 2.0 >= 0.0);
        assert!(self.center_pos_x + self.size_x / 2.0 <= MODEL_LEN_X);
        assert!(self.center_pos_y - self.size_y / 2.0 >= 0.0);
        assert!(self.center_pos_y + self.size_y / 2.0 <= MODEL_LEN_Y);
    }
}
impl Assert for Ball {
    fn assert(&self) {
        assert!(self.center.x - self.radius >= 0.0);
        assert!(self.center.x + self.radius <= MODEL_LEN_X);
        assert!(self.center.y - self.radius >= 0.0);
        assert!(self.center.y + self.radius <= MODEL_LEN_Y);
    }
}
