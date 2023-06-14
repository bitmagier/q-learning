//! This is the Rust interface for the Tensorflow model `keras_model/q_learning_model1`.
//! That model is generated by the python script `keras_model/create_q_learning_model.py`

use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, Tensor};

use crate::ai::model::model_function::{ModelFunction1, ModelFunction3};
use crate::app::{FRAME_SIZE_X, FRAME_SIZE_Y};

const KERAS_MODEL_DIR: &str = "keras_model/q_learning_model_1";
#[allow(unused)]
const KERAS_MODEL_CHECKPOINT_FILE: &str = "keras_model/q_learning_model_1_ckpt/checkpoint";

/// series of frames to represent world state
pub const WORLD_STATE_NUM_FRAMES: usize = 4;

// 600x800 pixel (grey-scaled), series of `WORLD_STATE_FRAMES` frames
pub const WORLD_STATE_DIMENSION: &[u64] = &[FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64];

// pub const ACTION_SPACE: u8 = 3;
pub const BATCH_SIZE: u64 = 32;

pub struct QLearningTfModel1 {
    pub graph: Graph,
    pub bundle: SavedModelBundle,
    fn_predict_single: ModelFunction1,
    fn_train_model: ModelFunction3,
    fn_write_checkpoint: ModelFunction1,
    fn_read_checkpoint: ModelFunction1,
}

#[allow(unused)]
impl QLearningTfModel1 {
    pub fn init() -> Self {
        // we load the model as a graph from the path it was saved in
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(KERAS_MODEL_DIR);
        let mut graph = Graph::new();

        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, model_dir,
        ).expect("Can't load model");

        // One way to get output names via saved_model_cli:
        // saved_model_cli show --dir /path/to/saved-model/ --all

        let fn_predict_single = ModelFunction1::new(&graph, &bundle, "predict_single", "state", "action");

        let fn_train_model = ModelFunction3::new(&graph, &bundle, "train_model", "state_samples", "action_samples", "updated_q_values", "loss");

        let fn_write_checkpoint = ModelFunction1::new(&graph, &bundle, "write_checkpoint", "path", "path");

        let fn_read_checkpoint = ModelFunction1::new(&graph, &bundle, "read_checkpoint", "path", "dummy");

        QLearningTfModel1 {
            graph,
            bundle,
            fn_predict_single,
            fn_train_model,
            fn_write_checkpoint,
            fn_read_checkpoint,
        }
    }

    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [FRAME_SIZE_X, FRAME_SIZE_Y, WORLD_STATE_NUM_FRAMES]
    ///
    pub fn predict(
        &self,
        state: &Tensor<f32>,
    ) -> u8 {
        let r = self.fn_predict_single.apply(&self.bundle.session, state);
        let r = r[0];
        assert!((0_i64..2).contains(&r));
        r as u8
    }

    /// Performs a single training step using a a batch of data.
    /// Returns the model's loss
    ///
    /// # Arguments
    /// * `state_samples` Tensor [BATCH_SIZE, FRAME_SIZE_X, FRAME_SIZE_Y, WORLD_STATE_NUM_FRAMES]
    /// * `action_samples` Tensor [BATCH_SIZE, 1]
    /// * `updated_q_values` Tensor [BATCH_SIZE, 1]
    ///
    pub fn train(
        &self,
        state_samples: &Tensor<f32>,
        action_samples: &Tensor<i8>,
        updated_q_values: &Tensor<f32>,
    ) -> f32 {
        let r = self.fn_train_model.apply(&self.bundle.session, state_samples, action_samples, updated_q_values);
        r[0]
    }

    pub fn write_checkpoint(
        &self,
        path: &str,
    ) -> String {
        let r = self.fn_write_checkpoint.apply::<String, String>(&self.bundle.session, &Tensor::from(path.to_string()));
        r[0].clone()
    }

    pub fn read_checkpoint(
        &self,
        path: &str,
    ) {
        let r = self.fn_read_checkpoint.apply::<String, String>(&self.bundle.session, &Tensor::from(path.to_string()));
    }
}


#[cfg(test)]
mod test {
    use tensorflow::Tensor;

    use crate::ai::model::q_learning_tf_model1::{BATCH_SIZE, KERAS_MODEL_CHECKPOINT_FILE, QLearningTfModel1, WORLD_STATE_NUM_FRAMES};
    use crate::app::{FRAME_SIZE_X, FRAME_SIZE_Y};

    #[test]
    fn test_load_model() {
        QLearningTfModel1::init();
    }

    #[test]
    fn test_predict() {
        let model = QLearningTfModel1::init();
        let state = Tensor::new(&[FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64]);
        model.predict(&state);
    }

    #[test]
    fn test_train() {
        let model = QLearningTfModel1::init();
        let state_samples = Tensor::new(&[BATCH_SIZE, FRAME_SIZE_X as u64, FRAME_SIZE_Y as u64, WORLD_STATE_NUM_FRAMES as u64]);
        let action_samples = Tensor::new(&[BATCH_SIZE, 1]);
        let updated_q_values = Tensor::new(&[BATCH_SIZE, 1]);

        let _loss = model.train(&state_samples, &action_samples, &updated_q_values);
    }

    #[test]
    fn test_save_and_load_model_ckpt() {
        let model = QLearningTfModel1::init();
        let path = model.write_checkpoint(KERAS_MODEL_CHECKPOINT_FILE);
        log::debug!("saved model to '{}'", path);

        model.read_checkpoint(KERAS_MODEL_CHECKPOINT_FILE);
    }
}
