use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use anyhow::Result;
use console_engine::pixel;
use console_engine::pixel::Pixel;
use console_engine::screen::Screen;
use rand::Rng;
use tensorflow::Tensor;

use crate::ql::prelude::{Action, DebugVisualizer, Environment, ModelActionType, QlError, ToMultiDimArray};

#[derive(Clone)]
pub struct BallGameStateFrames {
    frames: VecDeque<BallGameState>,
    steps: usize,
}
impl BallGameStateFrames {
    pub fn now(&self) -> &BallGameState { self.frames.front().expect("should have one state") }
}

const NUM_FRAMES: usize = 3;
const MAX_STEPS: usize = 10;

/// A quite simple TestEnvironment simulating a ball game.
///
/// 3x3 field (y=0 north / y=2 south)
/// - One goal - on a random column on the north row
/// - One ball
///     - initially on a random column on the south row
///     - may be moved by an action one field intp one of the four directions
/// - Two obstacles - one in the center at (1,1) and the other one somewhere on one of the remaining free fields.
/// - Game goal: move the ball into the goal - each round one step into one of the available directions: (west, north, east or south)
///
/// This environment requires a q-learning model with:
/// - input dims: `[3,3,12]`  (3x3 pixel, 3 frames x 4 channels)
/// - out dims: `[5]`
/// - batch_size: 32
pub struct BallGameTestEnvironment {
    state: BallGameStateFrames,
}

impl BallGameTestEnvironment {
    pub fn new() -> Self {
        Self {
            state: BallGameStateFrames {
                frames: VecDeque::from([BallGameState::random_initial_state()]),
                steps: 0,
            },
        }
    }

    fn add_new_state(
        &mut self,
        state: BallGameState,
    ) {
        self.state.steps += 1;
        self.state.frames.push_front(state);
        if self.state.frames.len() > NUM_FRAMES {
            self.state.frames.pop_back();
            assert_eq!(self.state.frames.len(), NUM_FRAMES);
        }
    }

    #[cfg(test)]
    pub fn test_state_00_01_11_22() -> Self {
        Self {
            state: BallGameStateFrames {
                frames: VecDeque::from([BallGameState::test_state_00_01_11_22()]),
                steps: 0,
            },
        }
    }
}

impl Environment for BallGameTestEnvironment {
    type S = BallGameStateFrames;
    type A = BallGameAction;

    fn reset(&mut self) {
        self.state.frames.clear();
        self.state.frames.push_front(BallGameState::random_initial_state());
        self.state.steps = 0;
    }

    fn state(&self) -> &Self::S { &self.state }

    fn step(
        &mut self,
        action: Self::A,
    ) -> (&Self::S, f32, bool) {
        let mut state = self.state.now().clone();
        let r = state.do_move(action);
        self.add_new_state(state);

        if let MoveResult::Legal { done: true } = r {
            (self.state(), 5.0, true)
        } else if self.state.steps >= MAX_STEPS {
            (self.state(), -1.0, true)
        } else if let MoveResult::Legal { done: false } = r {
            (self.state(), -0.005, false)
        } else if let MoveResult::Illegal = r {
            (self.state(), -0.01, false)
        } else {
            unreachable!()
        }
    }

    fn total_reward_goal(&self) -> f32 { 4.5 }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BallGameState {
    /// [x,y]
    field: Field,
    ball_coord: (usize, usize),
}

impl BallGameState {
    fn random_initial_state() -> Self {
        let goal_coord: (usize, usize) = (rand::thread_rng().gen_range(0..3), 0);
        let ball_coord: (usize, usize) = (rand::thread_rng().gen_range(0..3), 2);
        // set one obstacle in the middle and the other one randomly
        let obstacle1_coord = (1, 1);
        let obstacle2_coord = loop {
            let c = (rand::thread_rng().gen_range(0..3), rand::thread_rng().gen_range(0..3));
            if c != goal_coord && c != ball_coord && c != obstacle1_coord {
                break c;
            }
        };

        let mut field = Field::default();
        field.set(goal_coord, Entry::Goal);
        field.set(ball_coord, Entry::Ball);
        field.set(obstacle1_coord, Entry::Obstacle);
        field.set(obstacle2_coord, Entry::Obstacle);

        BallGameState { field, ball_coord }
    }

    fn do_move(
        &mut self,
        action: BallGameAction,
    ) -> MoveResult {
        use BallGameAction::*;
        const VALID_TARGET_ENTRIES: [Entry; 2] = [Entry::Empty, Entry::Goal];
        let valid_target_coord = |x, y| VALID_TARGET_ENTRIES.contains(&self.field.get((x, y)));

        let (x, y) = self.ball_coord;
        let valid_target = match action {
            West if x > 0 && valid_target_coord(x - 1, y) => Some((x - 1, y)),
            North if y > 0 && valid_target_coord(x, y - 1) => Some((x, y - 1)),
            East if x < 2 && valid_target_coord(x + 1, y) => Some((x + 1, y)),
            South if y < 2 && valid_target_coord(x, y + 1) => Some((x, y + 1)),
            Nothing => Some((x, y)),
            _ => None,
        };

        match valid_target {
            None => MoveResult::Illegal,
            Some(c @ (x, y)) => {
                let done = self.field.get((x, y)) == Entry::Goal;
                self.field.set(self.ball_coord, Entry::Empty);
                self.field.set(c, Entry::Ball);
                self.ball_coord = c;
                MoveResult::Legal { done }
            }
        }
    }

    #[cfg(test)]
    fn test_state_00_01_11_22() -> Self {
        let mut field = Field::default();

        field.set((0, 0), Entry::Goal);
        field.set((0, 1), Entry::Obstacle);
        field.set((1, 1), Entry::Obstacle);
        field.set((2, 2), Entry::Ball);

        BallGameState { field, ball_coord: (2, 2) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Empty,
    Goal,
    Ball,
    Obstacle,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BallGameAction {
    Nothing,
    West,
    North,
    East,
    South,
}

impl Display for BallGameAction {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Action for BallGameAction {
    const ACTION_SPACE: ModelActionType = 5;

    fn numeric(&self) -> ModelActionType {
        use BallGameAction::*;
        match self {
            Nothing => 4,
            West => 0,
            North => 1,
            East => 2,
            South => 3,
        }
    }

    fn try_from_numeric(value: ModelActionType) -> Result<Self> {
        use BallGameAction::*;
        match value {
            4 => Ok(Nothing),
            0 => Ok(West),
            1 => Ok(North),
            2 => Ok(East),
            3 => Ok(South),
            _ => Err(QlError(format!("value {} out of range", value)).into()),
        }
    }
}

enum MoveResult {
    Illegal,
    Legal { done: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field([[Entry; 3]; 3]);

impl Field {
    fn set(
        &mut self,
        coord: (usize, usize),
        entry: Entry,
    ) {
        self.0[coord.0][coord.1] = entry
    }

    fn get(
        &self,
        coord: (usize, usize),
    ) -> Entry {
        self.0[coord.0][coord.1]
    }
}

impl Default for Field {
    fn default() -> Self { Field([[Entry::Empty; 3]; 3]) }
}

impl DebugVisualizer for BallGameStateFrames {
    fn one_line_info(&self) -> String {
        let state = self.now();
        let goal_pos_x = (0_usize..3).find(|&x| state.field.get((x, 0)) == Entry::Goal);
        let distance = match goal_pos_x {
            None => 0, // ball already on goal pos
            Some(goal_pos_x) => {
                let distance_x = (state.ball_coord.0 as isize - goal_pos_x as isize).abs();
                let distance_y = (state.ball_coord.1 - 0) as isize;
                distance_x + distance_y
            }
        };
        format!("BallGameField: Ball-goal-distance: {}", distance).to_string()
    }

    fn render_to_console(&self) -> Screen {
        let mut screen = Screen::new_empty(3, 3);
        screen.clear();

        let state = self.now();
        for y in 0..state.field.0.len() {
            for x in 0..state.field.0[0].len() {
                let pixel: Option<Pixel> = match state.field.get((x, y)) {
                    Entry::Empty => None,
                    Entry::Goal => Some(pixel::pxl('□')),
                    Entry::Ball => Some(pixel::pxl('●')),
                    Entry::Obstacle => Some(pixel::pxl('x')),
                };
                if let Some(pixel) = pixel {
                    screen.set_pxl(x as i32, y as i32, pixel);
                }
            }
        }
        screen
    }
}

/// channels used for ONE-HOT encoding the field-entry
const CHANNEL_EMPTY: u64 = 0;
const CHANNEL_GOAL: u64 = 1;
const CHANNEL_BALL: u64 = 2;
const CHANNEL_OBSTACLE: u64 = 3;

impl ToMultiDimArray<Tensor<f32>> for BallGameStateFrames {
    fn dims(&self) -> &[u64] { &[3_u64, 3_u64, 12_u64] }

    // TODO (maybe) eliminate that special function by using a batch of 1 instead (also adjust/merge TF model function)
    fn to_multi_dim_array(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[3_u64, 3_u64, 12_u64]);
        for (i_frame, frame) in self.frames.iter().enumerate() {
            for y in 0_u64..3 {
                for x in 0_u64..3 {
                    let channel = match frame.field.get((x as usize, y as usize)) {
                        Entry::Empty => CHANNEL_EMPTY,
                        Entry::Goal => CHANNEL_GOAL,
                        Entry::Ball => CHANNEL_BALL,
                        Entry::Obstacle => CHANNEL_OBSTACLE,
                    };
                    tensor.set(&[x, y, (i_frame * 4) as u64 + channel], 1.0);
                }
            }
        }
        tensor
    }

    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[N as u64, 3_u64, 3_u64, 12_u64]);
        for b in 0_u64..N as u64 {
            for (i_frame, frame) in batch[b as usize].frames.iter().enumerate() {
                for y in 0_u64..3 {
                    for x in 0_u64..3 {
                        let channel = match frame.field.get((x as usize, y as usize)) {
                            Entry::Empty => CHANNEL_EMPTY,
                            Entry::Goal => CHANNEL_GOAL,
                            Entry::Ball => CHANNEL_BALL,
                            Entry::Obstacle => CHANNEL_OBSTACLE,
                        };
                        tensor.set(&[b, x, y, (i_frame * 4) as u64 + channel], 1.0)
                    }
                }
            }
        }
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ballgame_environment() {
        let mut env = BallGameTestEnvironment::test_state_00_01_11_22();
        let initial_state = env.state().now().clone();
        let (state, reward, done) = env.step(BallGameAction::East);
        let state = state.now();
        assert_eq!(state.field, initial_state.field);
        assert_eq!(state.ball_coord, initial_state.ball_coord);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::South);
        let state = state.now();
        assert_eq!(state.field, initial_state.field);
        assert_eq!(state.ball_coord, initial_state.ball_coord);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::North);
        let state = state.now();
        assert_eq!(state.ball_coord, (2, 1));
        assert_eq!(state.field.get((2, 1)), Entry::Ball);
        assert_eq!(state.field.get((2, 2)), Entry::Empty);
        assert_eq!(state.field.get((1, 2)), Entry::Empty);
        assert_eq!(state.field.get((0, 2)), Entry::Empty);
        assert_eq!(state.field.get((1, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((0, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((2, 0)), Entry::Empty);
        assert_eq!(state.field.get((1, 0)), Entry::Empty);
        assert_eq!(state.field.get((0, 0)), Entry::Goal);
        assert!(reward <= 0.0);
        assert_eq!(done, false);

        let last_state = state.clone();
        let (state, _, _) = env.step(BallGameAction::West);
        let state = state.now();
        assert_eq!(state.field, last_state.field);
        assert_eq!(state.ball_coord, last_state.ball_coord);

        let (state, reward, done) = env.step(BallGameAction::East);
        let state = state.now();
        assert_eq!(state.field, last_state.field);
        assert_eq!(state.ball_coord, last_state.ball_coord);
        assert!(reward <= 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::North);
        let state = state.now();
        assert_eq!(state.ball_coord, (2, 0));
        assert_eq!(state.field.get((2, 1)), Entry::Empty);
        assert_eq!(state.field.get((2, 0)), Entry::Ball);
        assert!(reward <= 0.0);
        assert_eq!(done, false);

        let last_state = state.clone();
        let (state, reward, done) = env.step(BallGameAction::North);
        let state = state.now();
        assert_eq!(state.field, last_state.field);
        assert_eq!(state.ball_coord, last_state.ball_coord);
        assert!(reward <= 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::West);
        let state = state.now();
        assert!(reward <= 0.0);
        assert_eq!(done, false);
        assert_eq!(state.ball_coord, (1, 0));
        assert_eq!(state.field.get((2, 0)), Entry::Empty);
        assert_eq!(state.field.get((1, 0)), Entry::Ball);

        let last_state = state.clone();
        let (state, reward, done) = env.step(BallGameAction::North);
        let state = state.now();
        assert_eq!(state.field, last_state.field);
        assert_eq!(state.ball_coord, last_state.ball_coord);
        assert!(reward <= 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::West);
        let state = state.now();
        assert_eq!(state.ball_coord, (0, 0));
        assert_eq!(state.field.get((1, 0)), Entry::Empty);
        assert_eq!(state.field.get((0, 0)), Entry::Ball);
        assert_eq!(state.field.get((0, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((1, 1)), Entry::Obstacle);

        assert!(reward > env.total_reward_goal());
        assert_eq!(done, true)
    }
}
