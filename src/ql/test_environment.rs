#![cfg(test)]

use std::fmt::{Display, Formatter};

use plotters::prelude::{CoordTranslate, DrawingArea, DrawingBackend};
use rand::Rng;
use tensorflow::Tensor;

use crate::ql::prelude::{Action, DebugVisualizer, Environment, ModelActionType, ToMultiDimArray};

/// A quite simple TestEnvironment simulating a ball game.
///
/// 3x3 field (y=0 north / y=2 south)
/// - One goal - on a random column on the north row
/// - One ball 
///     - initially on a random column on the south row
///     - may be moved by an action one field intp one of the four directions
/// - Two obstacles - somewhere on one of the remaining free fields.
///     - To guarantee, that the game remains solvable, we put one obstacle in the middle row. The other one goes to any of the remaining free fields.   
/// - Game goal: move the ball into the goal - each round one step into one of the available directions: (west, north, east or south)
///
/// This environment Requires a q-learning model with spec: 3x3x3_4_32
/// - input dims: `[3,3,3]`
/// - out dims: `[4]`
/// - batch_size: 32
pub struct BallGameTestEnvironment {
    state: BallGameState,
}

#[derive(Clone)]
pub struct BallGameState {
    /// [x,y]
    field: Field,
    ball_coord: (usize, usize),
}

impl BallGameState {
    fn do_move(&mut self, action: BallGameGameAction) -> MoveResult {
        use BallGameGameAction::*;
        const VALID_TARGET_ENTRIES: [Entry; 2] = [Entry::Void, Entry::Goal];
        let valid_target_coord = |x, y| VALID_TARGET_ENTRIES.contains(&self.field.get((x, y)));
        let (x, y) = self.ball_coord;
        let valid_target = match action {
            Left if x > 0 && valid_target_coord(x - 1, y) => Some((x - 1, y)),
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
}

enum MoveResult {
    Illegal,
    Legal { done: bool },
}

#[derive(Clone)]
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
        let goal_pos_x = (0_usize..3).into_iter().find(|&x| self.field.get((x, 0)) == Entry::Goal).expect("Should find a goal");
        let distance_x = (self.ball_coord.0 as isize - goal_pos_x as isize).abs();
        let distance_y = (self.ball_coord.1 - 0) as isize;
        let distance = distance_x + distance_y;
        format!("BallGameField: Ball-goal-distance: {}", distance).to_string()
    }

    fn plot<DB: DrawingBackend, CT: CoordTranslate>(&self, drawing_area: &mut DrawingArea<DB, CT>) {
        todo!()
    }
}


impl BallGameTestEnvironment {
    pub fn new() -> Self {
        Self {
            state: Self::random_initial_state()
        }
    }

    fn random_initial_state() -> BallGameState {
        let goal_coord: (usize, usize) = (rand::thread_rng().gen_range(0..3), 0);
        let ball_coord: (usize, usize) = (rand::thread_rng().gen_range(0..3), 2);
        // set one obstacle on the middle row and the other one randomly
        let obstacle1_coord = (rand::thread_rng().gen_range(0..3), 1);
        let obstacle2_coord = {
            let taken = [goal_coord, ball_coord, obstacle1_coord];
            loop {
                let coord = (rand::thread_rng().gen_range(0..3), rand::thread_rng().gen_range(0..3));
                if !taken.contains(&coord) {
                    break coord;
                }
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Goal,
    Ball,
    Obstacle,
    Void,
}

#[derive(Debug, Clone, Copy)]
pub enum BallGameGameAction {
    Left,
    North,
    East,
    South,
}

impl Display for BallGameGameAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Action for BallGameGameAction {
    const ACTION_SPACE: ModelActionType = 4;

    fn numeric(&self) -> ModelActionType {
        use BallGameGameAction::*;
        match self {
            Left => 0,
            North => 1,
            East => 2,
            South => 3,
        }
    }

    fn try_from_numeric(value: ModelActionType) -> Result<Self, String> {
        use BallGameGameAction::*;
        match value {
            0 => Ok(Left),
            1 => Ok(North),
            2 => Ok(East),
            3 => Ok(South),
            _ => Err(format!("value {} out of range", value))
        }
    }
}

impl Environment for BallGameTestEnvironment {
    type S = BallGameState;
    type A = BallGameGameAction;

    fn reset(&mut self) {
        self.state = BallGameTestEnvironment::random_initial_state()
    }

    fn state(&self) -> &Self::S {
        &self.state
    }

    fn step(&mut self, action: Self::A) -> (&Self::S, f32, bool) {
        match self.state.do_move(action) {
            MoveResult::Illegal => (self.state(), -0.1, false),
            MoveResult::Legal { done } if done => (self.state(), 10.0, done),
            MoveResult::Legal { done } => (self.state(), -0.1, done),
        }
    }

    fn total_reward_goal(&self) -> f32 {
        10.0 - 4.0 * 0.1
    }
}

impl ToMultiDimArray<Tensor<f32>> for BallGameState {
    fn dims(&self) -> &[u64] {
        &[3_u64, 3_u64, 3_u64]
    }

    fn to_multi_dim_array(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[3_u64, 3_u64, 3_u64]);
        for y in 0..3 {
            for x in 0..3 {
                match self.field.get((x, y)) {
                    Entry::Goal => tensor.set(&[x as u64, y as u64, 0_u64], 1.0),
                    Entry::Ball => tensor.set(&[x as u64, y as u64, 1_u64], 1.0),
                    Entry::Obstacle => tensor.set(&[x as u64, y as u64, 2_u64], 1.0),
                    Entry::Void => (),
                }
            }
        }
        tensor
    }

    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Self; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[N as u64, 3_u64, 3_u64, 3_u64]);
        for b in 0..N {
            for y in 0..3 {
                for x in 0..3 {
                    match batch[b].field.get((x, y)) {
                        Entry::Goal => tensor.set(&[b as u64, x as u64, y as u64, 0_u64], 1.0),
                        Entry::Ball => tensor.set(&[b as u64, x as u64, y as u64, 1_u64], 1.0),
                        Entry::Obstacle => tensor.set(&[b as u64, x as u64, y as u64, 2_u64], 1.0),
                        Entry::Void => (),
                    }
                }
            }
        }
        tensor
    }
}
