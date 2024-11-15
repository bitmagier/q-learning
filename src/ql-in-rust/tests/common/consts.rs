use std::path::PathBuf;

use lazy_static::lazy_static;

pub const BATCH_SIZE: usize = 512;

#[rustfmt::skip]
lazy_static! {
    pub static ref CHECKPOINT_FILE_BASE: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_model/checkpoints/itest_ballgame_3x3x4_5_512");
}
