use std::time::Duration;

use crate::pong::{Coordinate, max_f32, min_f32, Vector2d};

pub const MODEL_GRID_LEN_X: f32 = 600.0;
pub const MODEL_GRID_LEN_Y: f32 = 800.0;

const SPACE_GRANULARITY: f32 = 0.001;

/// time granularity (TG)
pub const TIME_GRANULARITY: Duration = Duration::from_millis(40);


const PANEL_LEN_X: f32 = 40.0;
const PANEL_LEN_Y: f32 = 10.0;
const PANEL_CENTER_POS_Y: f32 = 770.0;

/// model grid len per time granularity
const PANEL_MAX_SPEED_PER_SECOND: f32 = 60.0;

const PANEL_CONTROL_ACCEL_PER_SECOND: f32 = 20.0;
/// slow down if not accelerated
const PANEL_SLOW_DOWN_ACCEL_PER_SECOND: f32 = 5.0;


const BRICK_EDGE_LEN: f32 = 25.0;
const BRICK_SPACING: f32 = 2.0;
const BRICK_ROWS: usize = 3;
const FIRST_BRICK_ROW_TOP_Y: f32 = 37.0;

const BALL_RADIUS: f32 = 10.0;
const BALL_SPEED_PER_SEC: f32 = 80.0;


pub struct PongMechanics {
    mechanic_state: GameState,
}

impl PongMechanics {
    pub fn new() -> Self {
        Self {
            mechanic_state: GameState::default(),
        }
    }

    pub fn initial_bricks() -> Vec<Brick> {
        fn create_brick(left_x: f32, upper_y: f32) -> Brick {
            Brick {
                lower_left: Coordinate::from((left_x, upper_y - BRICK_EDGE_LEN)),
                upper_right: Coordinate::from((left_x + BRICK_EDGE_LEN, upper_y)),
            }
        }

        let mut bricks = vec![];
        for row in 0..BRICK_ROWS {
            let mut left_x = 0.0;
            let upper_y = FIRST_BRICK_ROW_TOP_Y + row as f32 * (BRICK_EDGE_LEN + BRICK_SPACING);
            loop {
                let brick = create_brick(left_x, upper_y);
                if brick.upper_right.x >= MODEL_GRID_LEN_X {
                    break;
                } else {
                    left_x = brick.upper_right.x + BRICK_SPACING;
                    bricks.push(brick);
                }
            }
        }
        bricks
    }

    pub fn initial_ball() -> Ball {
        Ball {
            center_pos: Coordinate::from((MODEL_GRID_LEN_X / 2.0, MODEL_GRID_LEN_Y / 2.0)),
            radius: BALL_RADIUS,
            direction: Vector2d::from((-0.2, 1.0)),
            speed_per_sec: BALL_SPEED_PER_SEC,
        }
    }

    pub fn initial_panel() -> Panel {
        Panel {
            center_pos_x: MODEL_GRID_LEN_X / 2.0,
            center_pos_y: PANEL_CENTER_POS_Y,
            size_x: PANEL_LEN_X,
            size_y: PANEL_LEN_Y,
            speed_per_sec: 0.0,
        }
    }

    pub fn time_step(
        &mut self,
        input: GameInput,
    ) -> GameState {
        self.mechanic_state.panel.proceed();
        self.mechanic_state.ball.proceed(&self.mechanic_state.panel, &mut self.mechanic_state.bricks);
        self.mechanic_state.panel.process_input(input);
        self.mechanic_state.clone()
    }
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

#[derive(Copy, Clone, PartialEq, Eq)]
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
    pub speed_per_sec: f32,
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
        let max_distance = max_f32(self.surfaces.iter().map(|e| e.distance));
        let min_distance = min_f32(self.surfaces.iter().map(|e| e.distance));
        assert!(max_distance - min_distance < SPACE_GRANULARITY);
    }
}



impl Ball {
    /// physically move one time step forward
    /// TODO project new ball position and direction based on its old position, speed, direction and collisions with panel&panel-drag/left-wall/right-wall
    pub fn proceed(&mut self, panel: &Panel, bricks: &mut Vec<Brick>) {
        assert!(self.speed_per_sec > 0.0);
        self.direction.normalize();
        let move_vector = self.direction * self.speed_per_sec * TIME_GRANULARITY.as_secs_f32();
        self.proceed_with(move_vector, panel, bricks);
    }

    #[allow(unused)]
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
    pub speed_per_sec: f32,
}

impl Panel {
    pub fn upper_edge_y(&self) -> f32 {
        self.center_pos_y + self.size_y / 2.0
    }
}

impl Panel {
    /// calculate new panel move-vector based on input or slow-down
    pub fn process_input(&mut self, input: GameInput) {
        match input.control {
            PanelControl::None =>
                self.speed_per_sec = decrease_speed(self.speed_per_sec, PANEL_SLOW_DOWN_ACCEL_PER_SECOND),
            PanelControl::AccelerateLeft =>
                self.speed_per_sec = accelerate(self.speed_per_sec, -PANEL_CONTROL_ACCEL_PER_SECOND, PANEL_MAX_SPEED_PER_SECOND),
            PanelControl::AccelerateRight =>
                self.speed_per_sec = accelerate(self.speed_per_sec, PANEL_CONTROL_ACCEL_PER_SECOND, PANEL_MAX_SPEED_PER_SECOND),
            PanelControl::Exit => ()
        }
    }
}

impl Panel {
    /// physically move one time step forward; project new position according to movement
    pub fn proceed(&mut self) {
        let most_left_center_pos_x = (self.size_x / 2.0).round();
        let most_right_center_pos_x = (MODEL_GRID_LEN_X - (self.size_x / 2.0)).round();
        let potential_pos_x = self.center_pos_x + self.speed_per_sec * TIME_GRANULARITY.as_secs_f32();

        if potential_pos_x <= most_left_center_pos_x {
            self.center_pos_x = most_left_center_pos_x;
            self.speed_per_sec = 0.0;
        } else if potential_pos_x >= most_right_center_pos_x {
            self.center_pos_x = most_right_center_pos_x;
            self.speed_per_sec = 0.0;
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

fn granulate_speed(speed: f32) -> f32 {
    (speed * 1000.0).round() / 1000.0
}

/// speed: LEN per TP
/// break_acceleration: LEN per TP²  (a positive amount)
fn decrease_speed(speed_per_sec: f32, break_acceleration_per_sec: f32) -> f32 {
    assert!(break_acceleration_per_sec >= 0.0);
    if speed_per_sec > 0.0 {
        (granulate_speed(speed_per_sec - break_acceleration_per_sec)).max(0.0)
    } else if speed_per_sec < 0.0 {
        (granulate_speed(speed_per_sec + break_acceleration_per_sec)).max(0.0)
    } else {
        0.0
    }
}

/// positive or negative speed and acceleration
fn accelerate(speed_per_sec: f32, acceleration_per_sec: f32, speed_limit_abs: f32) -> f32 {
    assert!(!speed_limit_abs.is_sign_negative());

    let virtual_speed = speed_per_sec + acceleration_per_sec;
    let result_speed =
        if virtual_speed.abs() > speed_limit_abs {
            match virtual_speed.is_sign_positive() {
                true => speed_limit_abs,
                false => -speed_limit_abs
            }
        } else {
            virtual_speed
        };

    granulate_speed(result_speed)
}
