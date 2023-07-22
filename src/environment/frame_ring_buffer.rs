use image::{ImageBuffer, Luma};
use rand::Rng;

// Gray-scaled image
pub type GrayFrame = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Clone, Debug)]
pub struct FrameRingBuffer<const NUM_FRAMES: usize> {
    pub frame_size_x: usize,
    pub frame_size_y: usize,
    /// `NUM_FRAMES` most recent frames
    pub buffer: [GrayFrame; NUM_FRAMES],
    pub next_slot: usize,
}

impl<const NUM_FRAMES: usize> FrameRingBuffer<NUM_FRAMES> {
    pub fn new(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
            buffer: (0..NUM_FRAMES).map(|_| GrayFrame::new(frame_size_x as u32, frame_size_y as u32))
                .collect::<Vec<_>>().try_into().unwrap(),
            next_slot: 0,
        }
    }

    pub fn random(frame_size_x: usize, frame_size_y: usize) -> Self {
        Self {
            frame_size_x,
            frame_size_y,
            buffer: (0..NUM_FRAMES).map(|_| GrayFrame::from_fn(frame_size_x as u32, frame_size_y as u32, |_, _| Luma::from([rand::thread_rng().gen::<u8>()])))
                .collect::<Vec<_>>().try_into().unwrap(),
            next_slot: 0,
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


