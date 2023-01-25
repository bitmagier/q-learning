use std::time::Duration;

use crate::pong::game::{Ball, Brick, Coordinate, GameInput, GameState, MODEL_GRID_LEN_X, MODEL_GRID_LEN_Y, Panel, Pong, Vector2d};

/// 5 means 5x5 pixel
const BRICK_EDGE_LEN: f32 = 25.0;
const BRICK_SPACING: f32 = 2.0;
const BRICK_ROWS: usize = 3;
const FIRST_BRICK_ROW_TOP_Y: f32 = 37.0;

const BALL_RADIUS: f32 = 10.0;
const BALL_SPEED_PER_TP: f32 = 1.5;

const PANEL_LEN_X: f32 = 40.0;
const PANEL_LEN_Y: f32 = 10.0;
const PANEL_CENTER_POS_Y: f32 = 770.0;

/// time portion (TP) / time granularity
pub const MODEL_TIME_PORTION: Duration = Duration::from_millis(200);


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
            speed: BALL_SPEED_PER_TP,
        }
    }

    pub fn initial_panel() -> Panel {
        Panel {
            center_pos_x: MODEL_GRID_LEN_X / 2.0,
            center_pos_y: PANEL_CENTER_POS_Y,
            size_x: PANEL_LEN_X,
            size_y: PANEL_LEN_Y,
            move_vector_x: 0.0,
        }
    }
}

impl Pong for PongMechanics {
    fn time_step(
        &mut self,
        input: GameInput,
    ) -> GameState {
        self.mechanic_state.panel.proceeed();
        self.mechanic_state.ball.proceed(&self.mechanic_state.panel, &self.mechanic_state.bricks);
        self.mechanic_state.panel.process_input(input);
        self.mechanic_state.clone()
    }
}

// TODO merge into game
