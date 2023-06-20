use std::rc::Rc;

use image::{ImageBuffer, imageops, Rgb};

use crate::breakout::mechanics::{BreakoutMechanics, GameInput, PanelControl};
use crate::ql::frame_ring_buffer::FrameRingBuffer;
use crate::ql::model::environment::Environment;
use super::{FRAME_SIZE_X, FRAME_SIZE_Y, WORLD_STATE_NUM_FRAMES};

pub type Action = u8;
pub type State = FrameRingBuffer<WORLD_STATE_NUM_FRAMES>;
pub type Reward = f32;


pub trait BreakoutDrawer {
    fn draw(&self, game_state: &BreakoutMechanics) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
}

pub struct BreakoutEnvironment {
    mechanics: BreakoutMechanics,
    drawer: Box<dyn BreakoutDrawer>,
    frame_buffer: FrameRingBuffer<WORLD_STATE_NUM_FRAMES>,
}

impl BreakoutEnvironment {
    pub fn new(drawer: Box<dyn BreakoutDrawer>) -> Self {
        Self {
            mechanics: BreakoutMechanics::new(),
            drawer,
            frame_buffer: FrameRingBuffer::new(FRAME_SIZE_X, FRAME_SIZE_Y),
        }
    }
}

impl Environment for BreakoutEnvironment {
    type State = State;
    type Action = Action;
    type Reward = Reward;

    fn reset(&mut self) {
        self.mechanics = BreakoutMechanics::new();
        self.frame_buffer = FrameRingBuffer::new(FRAME_SIZE_X, FRAME_SIZE_Y)
    }

    fn no_action(&self) -> Self::Action {
        map_game_input_to_model_action(GameInput::action(PanelControl::None))
    }

    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, Self::Reward, bool) {
        let prev_score = self.mechanics.score;
        let game_input: GameInput = map_model_action_to_game_input(action);
        self.mechanics.time_step(game_input);

        let frame = self.drawer.draw(&self.mechanics);
        let frame = imageops::grayscale(&frame);
        self.frame_buffer.add(frame);

        let state = Rc::new(self.frame_buffer.clone());
        let reward = (self.mechanics.score - prev_score) as f32;
        let done = self.mechanics.finished;

        (state, reward, done)
    }
}

fn map_model_action_to_game_input(model_action: Action) -> GameInput {
    GameInput::action(
        match model_action {
            0 => PanelControl::None,
            1 => PanelControl::AccelerateLeft,
            2 => PanelControl::AccelerateRight,
            _ => panic!("model_action out of range")
        })
}

fn map_game_input_to_model_action(game_input: GameInput) -> Action {
    match game_input.control {
        PanelControl::None => 0,
        PanelControl::AccelerateLeft => 1,
        PanelControl::AccelerateRight => 2
    }
}
