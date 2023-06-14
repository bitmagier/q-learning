#![allow(unused)]
//! It's quite a challenge to implement this async realtime AI play/learner efficiently while still trying to figure out, how to feed the model.
//! So probably a more efficient and easier way would it be to create an DirectAiLearner directly connected to [BreakoutMechanics] + using a Drawer to generate the image data


use std::sync::Arc;
use std::sync::mpsc::Sender;

use image::{ImageBuffer, Luma, Pixel};
use lazy_static::lazy_static;
use tensorflow::Tensor;
use threadpool::ThreadPool;

use crate::ai::model::q_learning_tf_model1::{QLearningTfModel1, WORLD_STATE_NUM_FRAMES};
use crate::app::{FRAME_SIZE_X, FRAME_SIZE_Y};
use crate::breakout::mechanics::{GameInput, PanelControl};

fn map_model_action_to_game_input(model_action: u8) -> GameInput {
    GameInput::action(
        match model_action {
            0 => PanelControl::None,
            1 => PanelControl::AccelerateLeft,
            2 => PanelControl::AccelerateRight,
            _ => panic!("model_action out of range")
        })
}

type GrayFrame = ImageBuffer<Luma<u8>, Vec<u8>>;

#[allow(unused_must_use)]
struct FrameRingBuffer {
    /// four most recent frames
    buffer: [GrayFrame; WORLD_STATE_NUM_FRAMES as usize],
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

    // TODO: it should bring a performance benefit with an optimized implementation here, when we re-order the WORLD_STATE dimensions, so that WORLD_STATE_NUM_FRAMES comes first,
    //   but it is also quite likely, that there is a (yet unseen) negative functional aspect involved. So we go alongside the reference implementation for now.
    pub fn as_tensor(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64]);
        for hist in 0..=3 {
            let frame = &self.buffer[hist];
            debug_assert_eq!(frame.len(), (FRAME_SIZE_X * FRAME_SIZE_Y) as usize);
            for y in 0..FRAME_SIZE_Y {
                for x in 0..FRAME_SIZE_X {
                    let pixel = frame.get_pixel(x as u32, y as u32);
                    tensor.set(&[x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                }
            }
        }
        tensor
    }

    fn get(&self, steps_into_history: usize) -> &GrayFrame {
        assert!(steps_into_history < 4, "available steps into history: 0..3");
        let slot = match self.next_slot as isize - 1 - steps_into_history as isize {
            slot @ 0.. => slot as usize,
            slot @ _ => (slot + 4) as usize
        };
        &self.buffer[slot]
    }
}

impl Default for FrameRingBuffer {
    fn default() -> Self {
        Self {
            buffer: [GrayFrame::default(), GrayFrame::default(), GrayFrame::default(), GrayFrame::default()],
            next_slot: 0,
        }
    }
}

pub struct RealtimeAiLearner {
    frames: FrameRingBuffer,
    model: QLearningTfModel1,
    thread_pool: ThreadPool,
}

impl RealtimeAiLearner {
    pub fn new(action_sender: Sender<GameInput>) -> Self {
        // TODO create an action-thread (passing action_sender) to react on watched frames
        // TODO create a learning-thread to learn from following states after an action
        Self {
            frames: FrameRingBuffer::default(),
            model: QLearningTfModel1::init(),
            thread_pool: ThreadPool::new(1),
        }
    }

    pub fn watch_next_frame(&mut self, frame: ImageBuffer<Luma<u8>, Vec<u8>>) {
        self.frames.add(frame);
        // TODO feed action-tread
        // TODO feed learning thread
    }
}



#[cfg(test)]
mod test {
    use crate::ai::realtime_ai_learner::FrameRingBuffer;

    #[test]
    fn test_tensor_linear_storage_assumptions() {
        let b = FrameRingBuffer::default();
        assert_eq!(b.as_tensor().get_index(&[0, 0, 0]), 0);
        assert_eq!(b.as_tensor().get_index(&[0, 0, 1]), 1);
        assert_eq!(b.as_tensor().get_index(&[0, 0, 2]), 2);
        assert_eq!(b.as_tensor().get_index(&[0, 1, 0]), 4);
        assert_eq!(b.as_tensor().get_index(&[0, 1, 1]), 5);
        assert_eq!(b.as_tensor().get_index(&[1, 0, 0]), 800 * 4);
        assert_eq!(b.as_tensor().get_index(&[1, 1, 0]), 800 * 4 + 4);
    }
}
