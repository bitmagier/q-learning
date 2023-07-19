use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;

use console_engine::screen::Screen;
use image::{imageops, Pixel};
use tensorflow::Tensor;
use anyhow::Result;

use crate::environment::breakout::breakout_drawer::BreakoutDrawer;
use crate::environment::breakout::mechanics::{BreakoutMechanics, GameInput, PanelControl};
use crate::environment::util::frame_ring_buffer::FrameRingBuffer;
use crate::ql::prelude::{Action, DebugVisualizer, Environment, ModelActionType, QlError, ToMultiDimArray};


const WORLD_STATE_NUM_FRAMES: usize = 4;

/// BreakoutState
///
/// We have 3 different state representations:
/// 1. original state = BreakoutMechanics
/// 2. n frames of a realistic state visualization (like last n camera frames)
/// 3. Tensor representation
#[derive(Clone)]
pub struct BreakoutState {
    mechanics: BreakoutMechanics,
    frame_buffer: FrameRingBuffer<WORLD_STATE_NUM_FRAMES>,
    model_dims: [u64; 3],
}

impl Debug for BreakoutState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "BreakoutState with {:?}", &self.mechanics)
    }
}

impl ToMultiDimArray<Tensor<f32>> for BreakoutState {
    fn dims(&self) -> &[u64] {
        &self.model_dims
    }

    fn to_multi_dim_array(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&self.model_dims);
        for hist in 0..WORLD_STATE_NUM_FRAMES {
            let frame = &self.frame_buffer.buffer[hist];
            for y in 0..self.frame_buffer.frame_size_y as u32 {
                for x in 0..self.frame_buffer.frame_size_x as u32 {
                    let pixel = frame.get_pixel(x, y);
                    tensor.set(&[x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                }
            }
        }
        tensor
    }

    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32> {
        let frame_size_x = batch[0].frame_buffer.frame_size_x;
        let frame_size_y = batch[0].frame_buffer.frame_size_y;
        let mut dims = Vec::with_capacity(4);
        dims.push(N as u64);
        dims.extend_from_slice(&batch[0].model_dims);
        let mut tensor = Tensor::new(&dims);

        for (batch_num, &state) in batch.into_iter().enumerate() {
            for hist in 0..WORLD_STATE_NUM_FRAMES {
                let frame = &state.frame_buffer.buffer[hist];
                debug_assert_eq!(frame.len(), (frame_size_x * frame_size_y));
                for y in 0..frame_size_y as u32 {
                    for x in 0..frame_size_x as u32 {
                        let pixel = frame.get_pixel(x, y);
                        tensor.set(&[batch_num as u64, x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                    }
                }
            }
        }
        tensor
    }
}

impl DebugVisualizer for BreakoutState {
    fn one_line_info(&self) -> String {
        format!("Breakout [{} bricks, ball_pos: {:?}, panel_pos: {:?}]",
                self.mechanics.bricks.len(), self.mechanics.ball.shape.center, self.mechanics.panel.shape.center()).to_string()
    }

    fn render_to_console(&self) -> Screen {
        todo!()
    }
}

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

    fn try_from_numeric(value: ModelActionType) -> Result<Self> {
        match value {
            0 => Ok(BreakoutAction::None),
            1 => Ok(BreakoutAction::Left),
            2 => Ok(BreakoutAction::Right),
            _ => Err(QlError("value out of range".to_string()))?
        }
    }
}

impl Display for BreakoutAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}


pub struct BreakoutEnvironment {
    frame_size_x: usize,
    frame_size_y: usize,
    state: BreakoutState,
    drawer: BreakoutDrawer,
}

impl BreakoutEnvironment {
    pub fn new(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
            state: BreakoutState {
                mechanics: BreakoutMechanics::new(),
                frame_buffer: FrameRingBuffer::new(frame_size_x, frame_size_y),
                model_dims: [frame_size_x as u64, frame_size_y as u64, WORLD_STATE_NUM_FRAMES as u64],
            },
            drawer: BreakoutDrawer::new(frame_size_x, frame_size_y),
        }
    }

    fn map_model_action_to_game_input(model_action: BreakoutAction) -> GameInput {
        GameInput::action(
            match model_action {
                BreakoutAction::None => PanelControl::None,
                BreakoutAction::Left => PanelControl::AccelerateLeft,
                BreakoutAction::Right => PanelControl::AccelerateRight
            })
    }

    #[allow(dead_code)]
    fn map_game_input_to_model_action(game_input: GameInput) -> BreakoutAction {
        match game_input.control {
            PanelControl::None => BreakoutAction::None,
            PanelControl::AccelerateLeft => BreakoutAction::Left,
            PanelControl::AccelerateRight => BreakoutAction::Right
        }
    }
}

impl Environment for BreakoutEnvironment {
    type S = BreakoutState;
    type A = BreakoutAction;

    fn reset(&mut self) {
        self.state.mechanics = BreakoutMechanics::new();
        self.state.frame_buffer = FrameRingBuffer::new(self.frame_size_x, self.frame_size_y)
    }

    fn state(&self) -> &Self::S {
        &self.state
    }

    fn step(&mut self, action: BreakoutAction) -> (&BreakoutState, f32, bool) {
        let prev_score = self.state.mechanics.score;
        let game_input: GameInput = Self::map_model_action_to_game_input(action);
        self.state.mechanics.time_step(game_input);

        let frame = self.drawer.draw(&self.state.mechanics);
        let frame = imageops::grayscale(&frame);
        self.state.frame_buffer.add(frame);

        let state = self.state();
        let reward = (self.state.mechanics.score - prev_score) as f32;
        let done = self.state.mechanics.finished;

        (state, reward, done)
    }

    fn total_reward_goal(&self) -> u64 {
        // hanging the goal a little lower than the exact value to avoid obstructive blur effects introduced by float calculations
        (BreakoutMechanics::new().bricks.len() - 1) as u64
    }
}
