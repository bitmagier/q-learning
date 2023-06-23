use std::fmt::{Display, Formatter};
use std::rc::Rc;

use plotters::backend::{PixelFormat, RGBPixel};
use plotters::prelude::*;
use rand::Rng;
use tensorflow::Tensor;

use q_learning_breakout::ql::model::q_learning_model1::{BATCH_SIZE, FRAME_SIZE_X, FRAME_SIZE_Y, ModelActionType, WORLD_STATE_NUM_FRAMES};
use q_learning_breakout::ql::prelude::{Action, Environment, State};

#[derive(Clone, Default)]
pub struct SlotThrowState {
    frame_size_x: isize,
    frame_size_y: isize,
    slot_middle_pos_x: isize,
    slot_with: isize,
    slot_move_vector: isize,
    player_middle_pos_x: isize,
}

impl SlotThrowState {
    fn step(&mut self) {
        (self.slot_middle_pos_x, self.slot_move_vector) =
            match self.slot_middle_pos_x + self.slot_move_vector {
                x if x < 0 => (-x, -self.slot_move_vector),
                x if x >= self.frame_size_x => (self.frame_size_x - x, -self.slot_move_vector),
                x => (x, self.slot_move_vector)
            }
    }

    fn hit_position(&self) -> bool {
        let range = (self.slot_middle_pos_x - self.slot_with / 4) .. (self.slot_middle_pos_x + self.slot_with / 4);
        range.contains(&(self.player_middle_pos_x))
    }

    fn draw_to_buffer(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // see (plotters)[https://github.com/plotters-rs/plotters]
        let mut buffer = vec![0; RGBPixel::PIXEL_SIZE * (self.frame_size_x * self.frame_size_y) as usize];
        {
            let drawing_area = BitMapBackend::with_buffer(buffer.as_mut_slice(), (self.frame_size_x as u32, self.frame_size_y as u32))
                .into_drawing_area();


            drawing_area.fill(&BLACK)?;

            let elements = || {
                assert!(self.slot_middle_pos_x < self.frame_size_x);
                assert!(self.player_middle_pos_x < self.frame_size_x);
                let wall_thickness = 40;
                EmptyElement::at((0, 0))
                    + Rectangle::new([(0, 0), (i32::max(0, self.slot_middle_pos_x as i32 - (self.slot_with / 2) as i32), wall_thickness)],
                                     ShapeStyle::from(&WHITE).filled())
                    + Rectangle::new([(i32::max(self.frame_size_x as i32 - 1, self.slot_middle_pos_x as i32 + (self.slot_with / 2) as i32), 0),
                                         (self.frame_size_x as i32 - 1, wall_thickness)],
                                     ShapeStyle::from(&WHITE).filled())
                    + Circle::new((0, 0), 20, ShapeStyle::from(&WHITE).stroke_width(3))
            };

            drawing_area.draw(&elements())?;
            drawing_area.present()?;
        }
        Ok(buffer)
    }
}

impl State for SlotThrowState {
    fn to_tensor(&self) -> Tensor<f32> {
        let image = self.draw_to_buffer().unwrap();
        let mut tensor = Tensor::<f32>::new(&[FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64]);

        let mut image_iter = image.chunks(RGBPixel::PIXEL_SIZE)
            .map(|c| ((c[0] as u16 + c[1] as u16 + c[2] as u16) / 3) as u8);

        for y in 0..self.frame_size_y {
            for x in 0..self.frame_size_x {
                let pixel = image_iter.next().unwrap();
                for f in 0..3 {
                    tensor.set(&[x as u64, y as u64, f as u64], pixel as f32)
                }
            }
        }
        tensor
    }

    fn batch_to_tensor<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[BATCH_SIZE as u64, FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64]);
        for (b, &state) in batch.iter().enumerate() {
            let image = state.draw_to_buffer().unwrap();
            let mut image_iter = image.chunks(RGBPixel::PIXEL_SIZE)
                .map(|c| ((c[0] as u16 + c[1] as u16 + c[2] as u16) / 3) as u8);

            for y in 0..state.frame_size_y {
                for x in 0..state.frame_size_x {
                    let pixel = image_iter.next().unwrap();
                    for f in 0..3 {
                        tensor.set(&[b as u64, x as u64, y as u64, f as u64], pixel as f32)
                    }
                }
            }
        }
        tensor
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SlotThrowAction {
    None,
    Throw,
}

impl Display for SlotThrowAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Action for SlotThrowAction {
    const ACTION_SPACE: ModelActionType = 2;

    fn numeric(&self) -> ModelActionType {
        match self {
            SlotThrowAction::None => 0,
            SlotThrowAction::Throw => 1
        }
    }

    fn try_from_numeric(value: ModelActionType) -> Result<Self, String> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Throw),
            _ => Err("value should be in range".to_string())
        }
    }
}

/// Simple simulated Test Environment
/// There is a wall on the upper screen with one moving slot left out.
/// There is a static player in the middle of the lower end of the screen pointing straight north (upwards) to throw a ball.
/// There are only 2 possible actions: do nothing or throw.
/// The goal is to hit the slot with every throw if possible.
/// ---
/// ```
/// ========  =====
///
///
///       °
/// ```
/// ---
pub(crate) struct SlotThrowEnvironment {
    state: SlotThrowState,
    steps: usize,
}

impl SlotThrowEnvironment {
    pub fn new() -> Self {
        let mut env = Self {
            state: SlotThrowState::default(),
            steps: 0,
        };
        env.reset();
        env
    }
}

impl Environment for SlotThrowEnvironment {
    type State = SlotThrowState;
    type Action = SlotThrowAction;

    fn reset(&mut self) {
        self.state = SlotThrowState {
            frame_size_x: FRAME_SIZE_X as isize,
            frame_size_y: FRAME_SIZE_Y as isize,
            slot_middle_pos_x: rand::thread_rng().gen_range(50..550),
            slot_with: 30,
            slot_move_vector: 5,
            player_middle_pos_x: rand::thread_rng().gen_range(200..480),
        };
        self.steps = 0
    }

    fn no_action() -> Self::Action {
        SlotThrowAction::None
    }

    fn step(&mut self, action: Self::Action) -> (Rc<Self::State>, f32, bool) {
        match action {
            SlotThrowAction::None => {
                let state = Rc::new(self.state.clone());
                let reward = 0.0;
                let done = false;
                (state, reward, done)
            }
            SlotThrowAction::Throw => {
                let (reward, done) = match action {
                    SlotThrowAction::None => {
                        (-0.001, false)
                    }
                    SlotThrowAction::Throw => {
                        if self.state.hit_position() {
                            (100.0, true)
                        } else {
                            (-20.0, false)
                        }
                    }
                };
                self.state.step();
                let state = Rc::new(self.state.clone());
                (state, reward, done)
            }
        }
    }

    fn total_reward_goal() -> f32 {
        90.0
    }
}

