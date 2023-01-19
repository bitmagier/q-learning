use std::time::Duration;

use crate::pong::game_api::{Ball, Brick, Coordinate, GameInput, GameState, Panel, Pong, Vector2d};

pub const BOARD_DIM_X: f32 = 600.0;
pub const BOARD_DIM_Y: f32 = 800.0;

/// 5 means 5x5 pixel
const BRICK_EDGE_LEN: f32 = 25.0;
const BRICK_SPACING: f32 = 2.0;
const BRICK_ROWS: usize = 3;
const FIRST_BRICK_ROW_TOP_Y: f32 = 763.0;

const BALL_RADIUS: f32 = 10.0;
const BALL_SPEED_PER_TP: f32 = 1.5;

const PANEL_SIZE_X: f32 = 40.0;

/// time portion (TP)
pub const TIME_GRANULARITY: Duration = Duration::from_millis(200);

/// pixel per time portion
const PANEL_MAX_SPEED_PER_TP: f32 = 1.0;
/// pixel per TP per TP
const PANEL_POSSIBLE_ACCELERATION_PER_TP: f32 = 0.1;
/// slow down if not accelerated (+/-)
const PANEL_SLOW_DOWN_PER_TP: f32 = 0.1;


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
        fn create_brick(left_x: f64, upper_y: f64) -> Brick {
            Brick {
                lower_left: Coordinate::from((left_x, upper_y - BRICK_EDGE_LEN)),
                upper_right: Coordinate::from((left_x + BRICK_EDGE_LEN, upper_y)),
            }
        }

        let mut bricks = vec![];
        for row in 0..BRICK_ROWS {
            let mut left_x: f64 = 0.0;
            let upper_y: f64 = FIRST_BRICK_ROW_TOP_Y + row as f64 * BRICK_EDGE_LEN + BRICK_SPACING;
            loop {
                let brick = create_brick(left_x, upper_y);
                if brick.upper_right.x >= BOARD_DIM_X {
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
            center: Coordinate::from((BOARD_DIM_X / 2.0, BOARD_DIM_Y / 2.0)),
            radius: BALL_RADIUS,
            direction: Vector2d::from((-0.2, 1.0)),
            speed: BALL_SPEED_PER_TP,
        }
    }

    pub fn initial_panel() -> Panel {
        Panel {
            center_pos_x: BOARD_DIM_X / 2.0,
            speed: 0.0,
        }
    }
}

impl Pong for PongMechanics {
    fn time_step(&mut self, _input: GameInput) -> GameState {
        // TODO
        self.mechanic_state.clone()
    }
}
