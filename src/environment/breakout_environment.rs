use std::fmt::{Display, Formatter};
use std::rc::Rc;

use image::imageops;

use crate::environment::breakout::breakout_drawer::BreakoutDrawer;
use crate::environment::breakout::mechanics::{BreakoutMechanics, GameInput, PanelControl};
use crate::environment::util::frame_ring_buffer::FrameRingBuffer;
use crate::ql::prelude::{Action, Environment, ModelActionType};


const FRAME_SIZE_X: usize = 600;
const FRAME_SIZE_Y: usize = 600;
const WORLD_STATE_NUM_FRAMES: usize = 4;

pub type BreakoutState = FrameRingBuffer<WORLD_STATE_NUM_FRAMES>;

#[derive(Debug, Clone, Copy)]
pub enum BreakoutAction {
    None,
    Left,
    Right,
}

impl Action for BreakoutAction {
    const ACTION_SPACE: ModelActionType = 3;

    fn numeric(&self) -> ModelActionType {
        match self {
            BreakoutAction::None => 0,
            BreakoutAction::Left => 1,
            BreakoutAction::Right => 2,
        }
    }

    fn try_from_numeric(value: ModelActionType) -> Result<Self, String> {
        match value {
            0 => Ok(BreakoutAction::None),
            1 => Ok(BreakoutAction::Left),
            2 => Ok(BreakoutAction::Right),
            _ => Err("value out of range".to_string())
        }
    }
}

impl Display for BreakoutAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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
    type State = BreakoutState;
    type Action = BreakoutAction;

    fn reset(&mut self) {
        self.mechanics = BreakoutMechanics::new();
        self.frame_buffer = FrameRingBuffer::new(FRAME_SIZE_X, FRAME_SIZE_Y)
    }

    fn no_action() -> Self::Action {
        Self::map_game_input_to_model_action(GameInput::action(PanelControl::None))
    }

    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, f32, bool) {
        let prev_score = self.mechanics.score;
        let game_input: GameInput = Self::map_model_action_to_game_input(action);
        self.mechanics.time_step(game_input);

        let frame = self.drawer.draw(&self.mechanics);
        let frame = imageops::grayscale(&frame);
        self.frame_buffer.add(frame);

        let state = Rc::new(self.frame_buffer.clone());
        let reward = (self.mechanics.score - prev_score) as f32;
        let done = self.mechanics.finished;

        (state, reward, done)
    }

    fn total_reward_goal() -> f32 {
        // hanging the goal a little lower than the exact value because of float calc / compare blur effects
        (BreakoutMechanics::new().bricks.len() as f32) - 0.01
    }
}

impl BreakoutEnvironment {
    fn map_model_action_to_game_input(model_action: <BreakoutEnvironment as Environment>::Action) -> GameInput {
        GameInput::action(
            match model_action {
                BreakoutAction::None => PanelControl::None,
                BreakoutAction::Left => PanelControl::AccelerateLeft,
                BreakoutAction::Right => PanelControl::AccelerateRight
            })
    }

    fn map_game_input_to_model_action(game_input: GameInput) -> <BreakoutEnvironment as Environment>::Action {
        match game_input.control {
            PanelControl::None => BreakoutAction::None,
            PanelControl::AccelerateLeft => BreakoutAction::Left,
            PanelControl::AccelerateRight => BreakoutAction::Right
        }
    }
}
