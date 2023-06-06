#![allow(unused)]

use futures::channel::mpsc::Sender;
use image::{ImageBuffer, Luma, Pixel};
use lazy_static::lazy_static;
use tensorflow::Tensor;

use crate::ai::model::q_learning_tf_model1::{QLearningTfModel1, WORLD_STATE_NUM_FRAMES};
use crate::app::{FRAME_SIZE_X, FRAME_SIZE_Y};
use crate::breakout::mechanics::{GameInput, PanelControl};

lazy_static!(
    pub static ref MODEL_ACTION_MAPPING: Vec<(PanelControl, u8)> = vec![
        (PanelControl::None, 0_u8),
        (PanelControl::AccelerateLeft, 1),
        (PanelControl::AccelerateRight, 2)
    ];
);


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

pub struct RealtimeAiPlayer {
    frames: FrameRingBuffer,
    model: QLearningTfModel1,
    action_sender: Sender<GameInput>,
}

impl RealtimeAiPlayer {
    pub fn new(action_sender: Sender<GameInput>) -> Self {
        Self {
            frames: FrameRingBuffer::default(),
            model: QLearningTfModel1::init(),
            action_sender,
        }
    }

    pub fn watch_next_frame(&mut self, frame: ImageBuffer<Luma<u8>, Vec<u8>>) {
        self.frames.add(frame);
        // TODO run model predict function if no such one is already running
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::ai::realtime_ai_player::FrameRingBuffer;

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
