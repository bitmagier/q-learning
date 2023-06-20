pub mod q_learning_tf_model1;
pub mod environment;
pub mod breakout_environment;
mod model_function;


// 600x800 pixel (grey-scaled), series of `WORLD_STATE_FRAMES` frames
pub const FRAME_SIZE_X: usize = crate::app::FRAME_SIZE_X;
pub const FRAME_SIZE_Y: usize = crate::app::FRAME_SIZE_Y;
pub const ACTION_SPACE: u8 = 3;

/// series of frames to represent world state
pub const WORLD_STATE_NUM_FRAMES: usize = 4;
pub const BATCH_SIZE: usize = 32;

