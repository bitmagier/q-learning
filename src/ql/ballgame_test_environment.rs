use std::fmt::{Display, Formatter};
use std::rc::Rc;

use anyhow::Result;
use console_engine::pixel;
use console_engine::pixel::Pixel;
use console_engine::screen::Screen;
use rand::Rng;
use tensorflow::Tensor;

use crate::ql::prelude::{Action, DebugVisualizer, Environment, ModelActionType, QlError, ToMultiDimArray};

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
/// This environment Requires a q-learning model with spec: 5x5x3_4_32
/// - input dims: `[5,5,3]`
/// - out dims: `[4]`
/// - batch_size: 32
pub struct BallGameTestEnvironment {
    state: BallGameState,
}

impl BallGameTestEnvironment {
    pub fn new() -> Self {
        Self {
            state: BallGameState::random_initial_state()
        }
    }

    #[cfg(test)]
    pub fn test_state_00_01_11_22() -> Self {
        Self {
            state: BallGameState::test_state_00_01_11_22()
        }
    }
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

        BallGameState {
            field,
            ball_coord,
        }
    }

    fn do_move(&mut self, action: BallGameAction) -> MoveResult {
        use BallGameAction::*;
        const VALID_TARGET_ENTRIES: [Entry; 2] = [Entry::Void, Entry::Goal];
        let valid_target_coord = |x, y| VALID_TARGET_ENTRIES.contains(&self.field.get((x, y)));
        
        let (x, y) = self.ball_coord;
        let valid_target = match action {
            West if x > 0 && valid_target_coord(x - 1, y) => Some((x - 1, y)),
            North if y > 0 && valid_target_coord(x, y - 1) => Some((x, y - 1)),
            East if x < 2 && valid_target_coord(x + 1, y) => Some((x + 1, y)),
            South if y < 2 && valid_target_coord(x, y + 1) => Some((x, y + 1)),
            _ => None
        };
        
        match valid_target {
            None => MoveResult::Illegal,
            Some(c @ (x, y)) => {
                let done = self.field.get((x, y)) == Entry::Goal;
                self.field.set(self.ball_coord, Entry::Void);
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

        BallGameState {
            field,
            ball_coord: (2, 2),
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
    fn set(&mut self, coord: (usize, usize), entry: Entry) {
        self.0[coord.0][coord.1] = entry
    }

    fn get(&self, coord: (usize, usize)) -> Entry {
        self.0[coord.0][coord.1]
    }
}

impl Default for Field {
    fn default() -> Self {
        Field([[Entry::Void; 3]; 3])
    }
}

impl DebugVisualizer for BallGameState {
    fn one_line_info(&self) -> String {
        let goal_pos_x = (0_usize..3).into_iter().find(|&x| self.field.get((x, 0)) == Entry::Goal);
        let distance = match goal_pos_x {
            None => 0, // ball already on goal pos
            Some(goal_pos_x) => {
                let distance_x = (self.ball_coord.0 as isize - goal_pos_x as isize).abs();
                let distance_y = (self.ball_coord.1 - 0) as isize;
                distance_x + distance_y
            }
        };
        format!("BallGameField: Ball-goal-distance: {}", distance).to_string()
    }

    fn render_to_console(&self) -> Screen {
        let mut screen = Screen::new_empty(3, 3);
        screen.clear();

        for y in 0..self.field.0.len() {
            for x in 0..self.field.0[0].len() {
                let pixel: Option<Pixel> = match self.field.get((x, y)) {
                    Entry::Goal => Some(pixel::pxl('□')),
                    Entry::Ball => Some(pixel::pxl('●')),
                    Entry::Obstacle => Some(pixel::pxl('x')),
                    Entry::Void => None
                };
                if let Some(pixel) = pixel {
                    screen.set_pxl(x as i32, y as i32, pixel);
                }
            }
        }
        screen
    }
}


#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Goal,
    Ball,
    Obstacle,
    Void,
}

#[derive(Debug, Clone, Copy)]
pub enum BallGameAction {
    West,
    North,
    East,
    South,
}

impl Display for BallGameAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Action for BallGameAction {
    const ACTION_SPACE: ModelActionType = 4;

    fn numeric(&self) -> ModelActionType {
        use BallGameAction::*;
        match self {
            West => 0,
            North => 1,
            East => 2,
            South => 3,
        }
    }

    fn try_from_numeric(value: ModelActionType) -> Result<Self> {
        use BallGameAction::*;
        match value {
            0 => Ok(West),
            1 => Ok(North),
            2 => Ok(East),
            3 => Ok(South),
            _ => Err(QlError(format!("value {} out of range", value)).into())
        }
    }
}

impl Environment for BallGameTestEnvironment {
    type S = BallGameState;
    type A = BallGameAction;

    fn reset(&mut self) {
        self.state = BallGameState::random_initial_state()
    }

    fn state(&self) -> &Self::S {
        &self.state
    }

    fn step(&mut self, action: Self::A) -> (&Self::S, f32, bool) {
        match self.state.do_move(action) {
            MoveResult::Illegal => (self.state(), -0.5, false),
            MoveResult::Legal { done } if done => (self.state(), 10.0, done),
            MoveResult::Legal { done } => (self.state(), -0.1, done),
        }
    }

    fn total_reward_goal(&self) -> u64 {
        9
    }
}


const CHANNEL_GOAL: u64 = 0;
const CHANNEL_BALL: u64 = 1;
const CHANNEL_OBSTACLE: u64 = 2;

impl ToMultiDimArray<Tensor<f32>> for BallGameState {
    fn dims(&self) -> &[u64] {
        &[5_u64, 5_u64, 3_u64]
    }

    /// out of a 3x3 field we create a 5x5 one with Obstacles at the border
    fn to_multi_dim_array(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[5_u64, 5_u64, 3_u64]);
        for x in 0..5 {
            tensor.set(&[x, 0, CHANNEL_OBSTACLE], 1.0);
            tensor.set(&[x, 4, CHANNEL_OBSTACLE], 1.0);
        }
        for y in 0..5 {
            tensor.set(&[0, y, CHANNEL_OBSTACLE], 1.0);
            tensor.set(&[4, y, CHANNEL_OBSTACLE], 1.0);
        }
        for y in 0_u64..3 {
            for x in 0_u64..3 {
                if let Some(channel) = match self.field.get((x as usize, y as usize)) {
                    Entry::Goal => Some(CHANNEL_GOAL),
                    Entry::Ball => Some(CHANNEL_BALL),
                    Entry::Obstacle => Some(CHANNEL_OBSTACLE),
                    Entry::Void => None,
                } {
                    tensor.set(&[x + 1, y + 1, channel], 1.0);
                }
            }
        }
        tensor
    }

    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[N as u64, 5_u64, 5_u64, 3_u64]);
        for b in 0_u64..N as u64 {
            for x in 0..5 {
                tensor.set(&[b, x, 0, CHANNEL_OBSTACLE], 1.0);
                tensor.set(&[b, x, 4, CHANNEL_OBSTACLE], 1.0);
            }
            for y in 0..5 {
                tensor.set(&[b, 0, y, CHANNEL_OBSTACLE], 1.0);
                tensor.set(&[b, 4, y, CHANNEL_OBSTACLE], 1.0);
            }
            for y in 0_u64..3 {
                for x in 0_u64..3 {
                    if let Some(channel) = match batch[b as usize].field.get((x as usize, y as usize)) {
                        Entry::Goal => Some(CHANNEL_GOAL),
                        Entry::Ball => Some(CHANNEL_BALL),
                        Entry::Obstacle => Some(CHANNEL_OBSTACLE),
                        Entry::Void => None,
                    } {
                        tensor.set(&[b, x + 1, y + 1, channel], 1.0)
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
        let initial_state = env.state().clone();
        let (state, reward, done) = env.step(BallGameAction::East);
        assert_eq!(*state, initial_state);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::South);
        assert_eq!(*state, initial_state);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::North);
        assert_ne!(*state, initial_state);
        assert_eq!(state.ball_coord, (2, 1));
        assert_eq!(state.field.get((2, 1)), Entry::Ball);
        assert_eq!(state.field.get((2, 2)), Entry::Void);
        assert_eq!(state.field.get((1, 2)), Entry::Void);
        assert_eq!(state.field.get((0, 2)), Entry::Void);
        assert_eq!(state.field.get((1, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((0, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((2, 0)), Entry::Void);
        assert_eq!(state.field.get((1, 0)), Entry::Void);
        assert_eq!(state.field.get((0, 0)), Entry::Goal);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let last_state = state.clone();

        let (state, reward, done) = env.step(BallGameAction::East);
        assert_eq!(*state, last_state);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::North);
        assert_eq!(state.ball_coord, (2, 0));
        assert_eq!(state.field.get((2, 1)), Entry::Void);
        assert_eq!(state.field.get((2, 0)), Entry::Ball);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let last_state = state.clone();

        let (state, reward, done) = env.step(BallGameAction::North);
        assert_eq!(*state, last_state);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::West);
        assert!(reward < 0.0);
        assert_eq!(done, false);
        assert_eq!(state.ball_coord, (1, 0));
        assert_eq!(state.field.get((2, 0)), Entry::Void);
        assert_eq!(state.field.get((1, 0)), Entry::Ball);

        let last_state = state.clone();

        let (state, reward, done) = env.step(BallGameAction::North);
        assert_eq!(*state, last_state);
        assert!(reward < 0.0);
        assert_eq!(done, false);

        let (state, reward, done) = env.step(BallGameAction::West);
        assert_eq!(state.ball_coord, (0, 0));
        assert_eq!(state.field.get((1, 0)), Entry::Void);
        assert_eq!(state.field.get((0, 0)), Entry::Ball);
        assert_eq!(state.field.get((0, 1)), Entry::Obstacle);
        assert_eq!(state.field.get((1, 1)), Entry::Obstacle);
        assert!(reward > 9.9 && reward < 10.1);
        assert_eq!(done, true)
    }
}
