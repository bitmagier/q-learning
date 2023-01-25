#![allow(unused)]

use std::ops::{Add, Neg};
use std::rc::Rc;
use std::time::Duration;

use egui::Vec2;

use crate::pong::pong_mechanics::PongMechanics;

pub const MODEL_GRID_LEN_X: f32 = 600.0;
pub const MODEL_GRID_LEN_Y: f32 = 800.0;

/// model grid len per time portion
const PANEL_MAX_SPEED_PER_TP: f32 = 1.0;

const PANEL_MASS_KG: f32 = 1.0;

const PANEL_CONTROL_ACCELERATION_LEN_PER_SQUARE_TP: f32 = 5.0;
/// slow down if not accelerated
const PANEL_SLOW_DOWN_ACCELERATION_LEN_PER_SQUARE_TP: f32 = 2.0;


#[derive(Copy, Clone)]
pub struct Vector2d {
    pub x: f32,
    pub y: f32,
}

impl Vector2d {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// normalize to len = 1.0
    pub fn normalize(&mut self) {
        let len = (self.x.powi(2) + self.y.powi(2)).sqrt();
        if (1.0 - len).abs() > 0.001 {
            let factor = 1.0 / len;
            self.x = self.x * factor;
            self.y = self.y * factor;
        }
    }
}

impl std::ops::Mul<f32> for Vector2d {
    type Output = Vector2d;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector2d {
            x: self.x * rhs,
            y: self.y * rhs
        }
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
            y: value.1 as f32,
        }
    }
}

pub type Coordinate = Vector2d;

impl Add<Vector2d> for Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: Vector2d) -> Self::Output {
        Coordinate {
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}

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
    pub center_pos: Coordinate,
    pub radius: f32,
    pub direction: Vector2d,
    pub speed: f32,
}

pub struct CollisionInfo {
    reflection_at_x: Option<f32>,
    reflection_at_y: Option<f32>,
}
impl CollisionInfo {
    pub fn at_x(x: f32) -> Self {
        CollisionInfo {
            reflection_at_x: Some(x),
            reflection_at_y: None
        }
    }

    pub fn at_y(y: f32) -> Self {
        CollisionInfo {
            reflection_at_x: None,
            reflection_at_y: Some(y),
        }
    }

    pub fn merge_with(&self, other: &Self) -> Self {
        fn merge(a: Option<f32>, b: Option<f32>) -> Option<f32> {
            match (a, b) {
                (x@Some(a), Some(b)) => {
                    assert!(roughly_equals(a, b));
                    x
                },
                (x@Some(a), None) => x,
                (None, x@Some(b)) => x,
                (None, None) => None
            }
        }
        Self {
            reflection_at_x: merge(self.reflection_at_x, other.reflection_at_x),
            reflection_at_y: merge(self.reflection_at_y, other.reflection_at_y)
        }
    }
}
impl Default for CollisionInfo {
    fn default() -> Self {
        Self {
            reflection_at_x: None,
            reflection_at_y: None,
        }
    }
}

impl Ball {
    /// physically move one time step forward
    /// TODO project new ball position and direction based on its old position, speed, direction and collisions with panel&panel-drag/left-wall/right-wall
    pub fn proceed(&mut self, panel: &Panel, bricks: &Vec<Brick>) {
        assert!(self.speed > 0.0);
        self.direction.normalize();
        let move_vector = self.direction * self.speed;
        self.proceed_with(move_vector, panel, bricks);
    }

    fn proceed_with(&mut self, move_vector: Vector2d, panel: &Panel, bricks: &Vec<Brick>) {
        let mut collision_info = CollisionInfo::default();

        if self.collision_with_left_wall(move_vector) {
            collision_info = collision_info.merge_with(&CollisionInfo::at_x(0.0))
        } else if self.collision_with_right_wall(move_vector) {
            collision_info = collision_info.merge_with(&CollisionInfo::at_x(MODEL_GRID_LEN_X))
        }

        if let Some(collision) = self.collision_y_from_north() {
            collision_info = collision_info.merge_with(&CollisionInfo::at_y(panel.upper_edge_y()));
            // TODO add influence of move_vector to effective reflection angle
        }

        // TODO calc point of reflection (keep ball canter pos)
        // TODO calc new ball move vector after reflection
        // TODO recursive call of proceed_with() with new move vector (different angle) and of remaining move length for TP,
    }
    
    fn collision_with_brick(&self, move_vector: Vector2d, brick: &Brick) -> Option<CollisionInfo> {
        todo!()
    }
    fn collision_with_left_wall(&self, move_vector: Vector2d) -> bool {
        todo!()
    }
    fn collision_with_right_wall(&self, move_vector: Vector2d) -> bool {
        todo!()
    }
    fn collision_y_from_north(&self) -> Option<CollisionInfo> {
        todo!()
    }
}

#[derive(Clone)]
pub struct Panel {
    pub center_pos_x: f32,
    pub center_pos_y: f32,
    pub size_x: f32,
    pub size_y: f32,
    /// model len per time portion
    pub move_vector_x: f32,
}

impl Panel {
    pub(crate) fn upper_edge_y(&self) -> f32 {
        self.center_pos_y + self.size_y / 2.0
    }
}

impl Panel {
    /// calculate new panel move-vector based on input or slow-down
    pub fn process_input(&mut self, input: GameInput) {
        match input.control {
            PanelControl::None => {
                self.move_vector_x = decrease_speed(self.move_vector_x, PANEL_SLOW_DOWN_ACCELERATION_LEN_PER_SQUARE_TP);
            }
            PanelControl::AccelerateLeft => {
                self.move_vector_x = accelerate(self.move_vector_x, PANEL_CONTROL_ACCELERATION_LEN_PER_SQUARE_TP, PANEL_MAX_SPEED_PER_TP);
            } PanelControl::AccelerateRight => {} PanelControl::Exit => {}
        }
    }
}

impl Panel {
    /// physically move one time step forward; project new position according to movement
    pub fn proceeed(&mut self) {
        let most_left_center_pos_x = (self.size_x / 2.0).round();
        let most_right_center_pos_x = (MODEL_GRID_LEN_X - (self.size_x / 2.0)).round();
        let potential_pos_x = self.center_pos_x + self.move_vector_x;

        if potential_pos_x <= most_left_center_pos_x {
            self.center_pos_x = most_left_center_pos_x;
            self.move_vector_x = 0.0;
        } else if potential_pos_x >= most_right_center_pos_x {
            self.center_pos_x = most_right_center_pos_x;
            self.move_vector_x = 0.0;
        } else {
            self.center_pos_x = potential_pos_x;
        }
    }
}

pub trait Assert {
    fn assert(&self);
}

impl Assert for Panel {
    fn assert(&self) {
        assert!(self.center_pos_x - self.size_x / 2.0 >= 0.0);
        assert!(self.center_pos_x + self.size_x / 2.0 <= MODEL_GRID_LEN_X);
        assert!(self.center_pos_y - self.size_y / 2.0 >= 0.0);
        assert!(self.center_pos_y + self.size_y / 2.0 <= MODEL_GRID_LEN_Y);
    }
}

impl Assert for Ball {
    fn assert(&self) {
        assert!(self.center_pos.x - self.radius >= 0.0);
        assert!(self.center_pos.x + self.radius <= MODEL_GRID_LEN_X);
        assert!(self.center_pos.y - self.radius >= 0.0);
        assert!(self.center_pos.y + self.radius <= MODEL_GRID_LEN_Y);
    }
}

fn roughly_equals(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.0001
}

fn granulate_coordinate(c: Coordinate) -> Coordinate {
    Coordinate {
        x: (c.x * 1000.0).round() / 1000.0,
        y: (c.y * 1000.0).round() / 1000.0
    }
}

fn granulate_speed(speed: f32) -> f32 {
    (speed * 100.0).round() / 100.0
}

/// speed: LEN per TP
/// break_acceleration: LEN per TP²  (a positive amount)
fn decrease_speed(speed: f32, break_acceleration: f32) -> f32 {
    assert!(break_acceleration >= 0.0);
    if speed > 0.0 {
        (granulate_speed(speed - break_acceleration)).max(0.0)
    } else if speed < 0.0 {
        (granulate_speed(speed + break_acceleration)).max(0.0)
    } else {
        0.0
    }
}

/// positive or negative acceleration
fn accelerate(speed: f32, acceleration: f32, speed_limit_abs: f32) -> f32 {
    assert!(!speed_limit_abs.is_sign_negative());
    let virtual_speed = speed + acceleration;
    let result =
        if virtual_speed.abs() > speed_limit_abs {
            match virtual_speed.is_sign_positive() {
                true => speed_limit_abs,
                false => speed_limit_abs.neg()
            }
        } else {
            virtual_speed
        };
    granulate_speed(result)
}
