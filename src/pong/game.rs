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

const SPACE_GRANULARITY: f32 = 0.001;

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
            y: self.y * rhs,
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
            y: self.y + rhs.y,
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

pub struct CollisionCandidates {
    surfaces: Vec<CollisionSurface>,
}

#[derive(Copy, Clone)]
pub struct CollisionSurface {
    // pub point: Coordinate,
    pub angle: f32,
    // 0 - 180 °
    pub distance: f32,
}

impl CollisionCandidates {
    pub fn new() -> Self {
        Self { surfaces: Vec::with_capacity(2) }
    }

    pub fn consider(&mut self, candidate: CollisionSurface) {
        assert!(!candidate.distance.is_sign_negative());
        if !self.surfaces.iter()
            .any(|&e| (e.distance < candidate.distance - SPACE_GRANULARITY)) {
            self.surfaces.push(candidate)
        }
    }

    pub fn effective_reflection(&self) -> Option<CollisionSurface> {
        match self.surfaces.len() {
            0 => None,
            1 => Some(*self.surfaces.first().unwrap()),
            _ => {
                self.multiple_entries_plausibility_check();
                let min_angle = min_f32(self.surfaces.iter().map(|e| e.angle));
                let max_angle: f32 = max_f32(self.surfaces.iter().map(|e| e.angle));

                let effective_angle = (min_angle + max_angle) / 2.0;
                Some(CollisionSurface {
                    angle: effective_angle,
                    distance: 0.0,
                })
            }
        }
    }

    fn multiple_entries_plausibility_check(&self) {
        let mut max_distance = max_f32(self.surfaces.iter().map(|e| e.distance));
        let mut min_distance = min_f32(self.surfaces.iter().map(|e| e.distance));
        assert!(max_distance - min_distance < SPACE_GRANULARITY);
    }
}

fn min_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::INFINITY, |a, b| a.min(b));
    assert!(r.is_finite());
    r
}

fn max_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::NEG_INFINITY, |a, b| a.max(b));
    assert!(r.is_finite());
    r
}



impl Ball {
    /// physically move one time step forward
    /// TODO project new ball position and direction based on its old position, speed, direction and collisions with panel&panel-drag/left-wall/right-wall
    pub fn proceed(&mut self, panel: &Panel, bricks: &mut Vec<Brick>) {
        assert!(self.speed > 0.0);
        self.direction.normalize();
        let move_vector = self.direction * self.speed;
        self.proceed_with(move_vector, panel, bricks);
    }

    fn proceed_with(&mut self, move_vector: Vector2d, panel: &Panel, bricks: &mut Vec<Brick>) {
        let mut collision_candidates = CollisionCandidates::new();
        // TODO test collisions and keep the one(s) with shortest distance
        // left + right wall
        // bricks
        // panel

        let collisions = collision_candidates;
        // TODO remove brick(s), which has been really hit

        if let Some(reflection) = collisions.effective_reflection() {
            // TODO calc point of reflection (keep ball canter pos)
            // TODO calc new ball move vector after reflection
            // TODO recursive call of proceed_with() with new move vector (different angle) and of remaining move length for TP,
        } else {
            self.center_pos = self.center_pos + move_vector
        }
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
    pub fn proceed(&mut self) {
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
        y: (c.y * 1000.0).round() / 1000.0,
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
