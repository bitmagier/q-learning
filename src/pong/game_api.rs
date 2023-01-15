#![allow(unused)]

use std::rc::Rc;

const BOARD_DIM_X: usize = 600;
const BOARD_DIM_Y: usize = 800;

/// 5 means 5x5 pixel
const BRICK_SIZE: usize = 5;
const BRICK_SPACING: usize = 1;

const PANEL_SIZE_X: usize = 20;

/// time portion (TP)
const TIME_GRANULARITY_IN_MS: usize = 200;

/// pixel per time portion
const PANEL_MAX_SPEED_PER_TP: f64 = 1.0;
/// pixel per TP per TP
const PANEL_POSSIBLE_ACCELERATION_PER_TP: f64 = 0.1;
/// slow down if not accelerated (+/-)
const PANEL_SLOW_DOWN_PER_TP: f64 = 0.1;


pub struct Coordinate(f64, f64);
pub struct Vector2d(f64, f64);

trait Pong {
    fn state() -> Rc<GameState>;
    fn time_step(input: PlayerInput);
}

pub struct GameState {
    // x = 0 = left side; y = 0 = bottom
    bricks: Vec<Brick>,
    ball: Ball,
    panel: Panel,
}

pub enum PlayerInput {
    Empty,
    AccelerateLeft,
    AccelerateRight
}

pub struct Brick {
    // 0,0 is in the lower left corner
    pub lower_left: Coordinate,
    pub upper_right: Coordinate
}

/// A ball is a perfect round 2D structure
pub struct Ball {
    pub center: Coordinate,
    pub radius: f64,
    direction: Vector2d,
    speed: f64,
}

pub struct Panel {
    pub center_pos_x: usize,
    speed: f64,
}
