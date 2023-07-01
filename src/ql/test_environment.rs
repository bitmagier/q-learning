#![cfg(test)]

use std::fmt::{Display, Formatter};
use std::rc::Rc;

use rand::Rng;
use tensorflow::Tensor;

use crate::ql::model::tensorflow::tf::TensorflowEnvironment;
use crate::ql::prelude::{Action, Environment, ModelActionType};

/// BallGameTestEnvironment
/// 3x3 field (y=0 north / y=2 south)
/// 1 ball - initially on a random column on the south row
/// 1 goal - on a random column on the north row
/// 2 obstacles - each blocking one field somewhere on the remaining free fields
/// Game  goal: move the ball into the goal - each round one step west, north, east or south.
///
/// Requires a q-learning model with input dimensions [3x3x3to4]
pub struct BallGameTestEnvironment {
    /// [x,y]
    state: Field,
    ball_coord: (usize, usize),
}

impl BallGameTestEnvironment {
    fn do_move(&mut self, action: BallGameGameAction) -> MoveResult {
        use BallGameGameAction::*;
        const VALID_TARGET_ENTRIES: [Entry; 2] = [Entry::Void, Entry::Goal];
        let valid_target_coord = |x, y| VALID_TARGET_ENTRIES.contains(&self.state.get((x, y)));
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
            Some(c@(x,y)) => {
                let finish = self.state.get((x,y)) == Entry::Goal;
                self.state.set(self.ball_coord, Entry::Void);
                self.state.set(c, Entry::Ball);
                self.ball_coord = c;
                MoveResult::Legal { finish }
            }
        }
    }
}

enum MoveResult {
    Illegal,
    Legal { finish: bool },
}

#[derive(Clone)]
pub struct Field([[Entry; 3]; 3]);

impl Default for Field {
    fn default() -> Self {
        Field([[Entry::Void; 3]; 3])
    }
}

impl Field {
    fn set(&mut self, coord: (usize, usize), entry: Entry) {
        self.0[coord.0][coord.1] = entry
    }

    fn get(&self, coord: (usize, usize)) -> Entry {
        self.0[coord.0][coord.1]
    }
}

impl BallGameTestEnvironment {
    pub fn new() -> Self {
        let (state, ball_coord) = Self::random_initial_state();
        Self {
            state,
            ball_coord,
        }
    }

    fn random_initial_state() -> (Field, (usize, usize)) {
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
        (field, ball_coord)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Entry {
    Void,
    Obstacle,
    Goal,
    Ball,
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
    type S = Field;
    type A = BallGameGameAction;

    fn reset(&mut self) {
        (self.state, self.ball_coord) = BallGameTestEnvironment::random_initial_state()
    }

    fn state(&self) -> Rc<Self::S> {
        Rc::new(self.state.clone())
    }

    fn step(&mut self, action: Self::A) -> (Rc<Self::S>, f32, bool) {
        match self.do_move(action) {
            MoveResult::Illegal => (self.state(), -0.1, false),
            MoveResult::Legal { finish } if finish => (self.state(), 10.0, finish),
            MoveResult::Legal { finish } => (self.state(), -0.1, finish),
        }
    }

    // given when reached the goal
    fn total_reward_goal(&self) -> f32 {
        10.0 - 4.0 * 0.1
    }
}

impl TensorflowEnvironment for BallGameTestEnvironment {
    fn state_dims(state: &Self::S) -> &[u64] {
        todo!()
    }

    fn state_to_tensor(state: &Self::S) -> Tensor<f32> {
        todo!()
    }

    fn state_batch_to_tensor<const BATCH_SIZE: usize>(batch: &[&Rc<Self::S>; BATCH_SIZE]) -> Tensor<f32> {
        todo!()
    }
}