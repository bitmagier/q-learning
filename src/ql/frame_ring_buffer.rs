use std::rc::Rc;
use image::{ImageBuffer, Luma, Pixel};
use tensorflow::Tensor;

use crate::app::{FRAME_SIZE_X, FRAME_SIZE_Y};
use crate::ql::model::q_learning_tf_model1::{BATCH_SIZE, WORLD_STATE_NUM_FRAMES};

pub type GrayFrame = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Clone)]
pub struct FrameRingBuffer {
    /// four most recent frames
    buffer: [GrayFrame; WORLD_STATE_NUM_FRAMES],
    next_slot: usize,
}

impl FrameRingBuffer {
    pub fn add(&mut self, element: GrayFrame) {
        self.buffer[self.next_slot] = element;
        self.next_slot = match self.next_slot + 1 {
            slot @ 1..=3 => slot,
            4 => 0,
            _ => panic!()
        }
    }

    fn _get(&self, steps_into_history: usize) -> &GrayFrame {
        assert!(steps_into_history < 4, "available steps into history: 0..3");
        let slot = match self.next_slot as isize - 1 - steps_into_history as isize {
            slot @ 0.. => slot as usize,
            slot @ _ => (slot + 4) as usize
        };
        &self.buffer[slot]
    }

    pub fn to_tensor(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[
            FRAME_SIZE_X as u64,
            FRAME_SIZE_Y as u64,
            WORLD_STATE_NUM_FRAMES as u64]
        );
        for hist in 0..WORLD_STATE_NUM_FRAMES {
            let frame = &self.buffer[hist];
            debug_assert_eq!(frame.len(), (FRAME_SIZE_X * FRAME_SIZE_Y));
            for y in 0..FRAME_SIZE_Y {
                for x in 0..FRAME_SIZE_X {
                    let pixel = frame.get_pixel(x as u32, y as u32);
                    tensor.set(&[x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                }
            }
        }
        tensor
    }

    pub fn batch_to_tensor<const N: usize>(batch: &[&Rc<FrameRingBuffer>; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[
            BATCH_SIZE as u64,
            FRAME_SIZE_X as u64,
            FRAME_SIZE_Y as u64,
            WORLD_STATE_NUM_FRAMES as u64
        ]);
        for (batch_num, state) in batch.into_iter().enumerate() {
            for hist in 0..WORLD_STATE_NUM_FRAMES {
                let frame = &state.buffer[hist];
                debug_assert_eq!(frame.len(), (FRAME_SIZE_X * FRAME_SIZE_Y));
                for y in 0..FRAME_SIZE_Y {
                    for x in 0..FRAME_SIZE_X {
                        let pixel = frame.get_pixel(x as u32, y as u32);
                        tensor.set(&[batch_num as u64, x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                    }
                }
            }
        }
        tensor
    }
}

impl Default for FrameRingBuffer {
    fn default() -> Self {
        Self {
            buffer: (0..4).map(|e| GrayFrame::new(FRAME_SIZE_X as u32, FRAME_SIZE_Y as u32))
                .collect::<Vec<_>>().try_into().unwrap(),
            next_slot: 0,
        }
    }
}
