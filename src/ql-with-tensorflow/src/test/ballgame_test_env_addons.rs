use std::rc::Rc;
use tensorflow::Tensor;
use ql::test::ballgame_test_environment::*;
use crate::ml_model::model::ToMultiDimArray;

/// channels used for ONE-HOT encoding the field-entry
const CHANNEL_EMPTY: u64 = 0;
const CHANNEL_GOAL: u64 = 1;
const CHANNEL_BALL: u64 = 2;
const CHANNEL_OBSTACLE: u64 = 3;


impl ToMultiDimArray<Tensor<f32>> for BallGameState {
    fn dims(&self) -> &[u64] { &[3_u64, 3_u64, 4_u64] }

    // TODO (maybe) eliminate that special function by using a batch of 1 instead (also adjust/merge TF model function)
    fn to_multi_dim_array(&self) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[3_u64, 3_u64, 4_u64]);
        for y in 0_u64..3 {
            for x in 0_u64..3 {
                let channel = match self.field.get((x as usize, y as usize)) {
                    Entry::Empty => CHANNEL_EMPTY,
                    Entry::Goal => CHANNEL_GOAL,
                    Entry::Ball => CHANNEL_BALL,
                    Entry::Obstacle => CHANNEL_OBSTACLE,
                };
                tensor.set(&[x, y, channel], 1.0);
            }
        }
        tensor
    }

    fn batch_to_multi_dim_array<const N: usize>(batch: &[&Rc<Self>; N]) -> Tensor<f32> {
        let mut tensor = Tensor::new(&[N as u64, 3_u64, 3_u64, 4_u64]);
        for (b, &state) in batch.iter().enumerate() {
            for y in 0_u64..3 {
                for x in 0_u64..3 {
                    let channel = match state.field.get((x as usize, y as usize)) {
                        Entry::Empty => CHANNEL_EMPTY,
                        Entry::Goal => CHANNEL_GOAL,
                        Entry::Ball => CHANNEL_BALL,
                        Entry::Obstacle => CHANNEL_OBSTACLE,
                    };
                    tensor.set(&[b as u64, x, y, channel], 1.0)
                }
            }
        }
        tensor
    }
}