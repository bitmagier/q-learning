use std::rc::Rc;

use image::{ImageBuffer, Luma, Pixel};
use rand::Rng;
use tensorflow::Tensor;

use crate::ql::model::q_learning_model::ToTensor;

// Gray-scaled image
pub type GrayFrame = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Clone, Debug)]
pub struct FrameRingBuffer<const NUM_FRAMES: usize> {
    frame_size_x: usize,
    frame_size_y: usize,
    /// `NUM_FRAMES` most recent frames
    buffer: [GrayFrame; NUM_FRAMES],
    next_slot: usize,
    model_dims: [u64; 3],
}

impl<const NUM_FRAMES: usize> FrameRingBuffer<NUM_FRAMES> {
    pub fn new(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
            buffer: (0..NUM_FRAMES).map(|_| GrayFrame::new(frame_size_x as u32, frame_size_y as u32))
                .collect::<Vec<_>>().try_into().unwrap(),
            next_slot: 0,
            model_dims: [frame_size_x as u64, frame_size_y as u64, NUM_FRAMES as u64],
        }
    }

    pub fn random(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
            buffer: (0..NUM_FRAMES).map(|_| GrayFrame::from_fn(frame_size_x as u32, frame_size_y as u32, |_, _| Luma::from([rand::thread_rng().gen::<u8>()])))
                .collect::<Vec<_>>().try_into().unwrap(),
            next_slot: 0,
            model_dims: [frame_size_x as u64, frame_size_y as u64, NUM_FRAMES as u64],
        }
    }

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
}


impl<const NUM_FRAMES: usize> ToTensor for FrameRingBuffer<NUM_FRAMES> {
    fn dims(&self) -> &[u64] {
        &self.model_dims
    }

    fn to_tensor(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[
            self.frame_size_x as u64,
            self.frame_size_y as u64,
            NUM_FRAMES as u64]
        );
        for hist in 0..NUM_FRAMES {
            let frame = &self.buffer[hist];
            for y in 0..self.frame_size_y as u32 {
                for x in 0..self.frame_size_x as u32 {
                    let pixel = frame.get_pixel(x, y);
                    tensor.set(&[x as u64, y as u64, hist as u64], pixel.channels()[0] as f32)
                }
            }
        }
        tensor
    }

    fn batch_to_tensor<const N: usize>(batch: &[&Rc<FrameRingBuffer<NUM_FRAMES>>; N]) -> Tensor<f32> {
        let frame_size_x = batch[0].frame_size_x;
        let frame_size_y = batch[0].frame_size_y;
        let mut tensor = Tensor::new(&[
            N as u64,
            frame_size_x as u64,
            frame_size_y as u64,
            NUM_FRAMES as u64
        ]);
        for (batch_num, &state) in batch.into_iter().enumerate() {
            for hist in 0..NUM_FRAMES {
                let frame = &state.buffer[hist];
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
