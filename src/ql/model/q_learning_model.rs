use std::path::Path;
use std::rc::Rc;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, Tensor};

use crate::ql::model::model_function::{ModelFunction1, ModelFunction3};
use crate::ql::prelude::{Action, Environment, ModelActionType, QLearningModel, State};

pub trait ToTensor {
    /// Diemsions, the object is represented towards the model.
    ///
    /// # Examples
    /// E.g we would use dimensions `[600,600,4]` for an environment state, which is represented 
    /// by a series of four grayscale frames with a frame size of 600x600.  
    fn dims(&self) -> &[u64];

    /// Produce a tensor with the dimensions returned by [Self::dims]  
    fn to_tensor(&self) -> Tensor<f32>;

    /// Produce a tensor of a batch of objects.
    /// The expected dimensionality of the returned tensor is one higher than returned by [Self::to_tensor], having `BATCH_SIZE` as the first axis.  
    fn batch_to_tensor<const BATCH_SIZE: usize>(batch: &[&Rc<Self>; BATCH_SIZE]) -> Tensor<f32>;
}


pub struct QLearningTensorflowModel {
    bundle: SavedModelBundle,
    fn_predict_single: ModelFunction1,
    fn_batch_predict_future_reward: ModelFunction1,
    fn_train_model: ModelFunction3<1>,
    fn_write_checkpoint: ModelFunction1,
    fn_read_checkpoint: ModelFunction1,
}

impl QLearningTensorflowModel {
    /// Init
    ///
    /// # Arguments
    /// * `model_dir` saved tensorflow model;  e.g. Path::new(env!("CARGO_MANIFEST_DIR")).join("tf_model/saved/q_learning_model_50x50x4to3")
    /// * `model_state_dim` input dimensions of the model; e.g. [600x600x4] describing four 600x600 grayscale frames; (used for input data versus model dim reference checks)
    /// * `action_space` number of possible actions = size of the model's one dimension output vector; (used for input data versus model dim reference checks)
    fn init(model_dir: &Path) -> Self {
        // we load the model as a graph from the path it was saved in
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, model_dir.to_str().expect("should have UTF-8 compatible path"),
        ).expect("Can't load model");

        // One way to get output names via saved_model_cli:
        // saved_model_cli show --dir /path/to/saved-model/ --all

        let fn_predict_single = ModelFunction1::new(&graph, &bundle, "predict_action", "state", "action");
        let fn_batch_predict_future_reward = ModelFunction1::new(&graph, &bundle, "batch_predict_future_reward", "state_batch", "reward_batch");
        let fn_train_model = ModelFunction3::new(&graph, &bundle, "train_model", "state_batch", "action_batch", "updated_q_values", &["loss"]);
        let fn_write_checkpoint = ModelFunction1::new(&graph, &bundle, "write_checkpoint", "file", "file");
        let fn_read_checkpoint = ModelFunction1::new(&graph, &bundle, "read_checkpoint", "file", "dummy");

        QLearningTensorflowModel {
            bundle,
            fn_predict_single,
            fn_batch_predict_future_reward,
            fn_train_model,
            fn_write_checkpoint,
            fn_read_checkpoint,
        }
    }

    fn check_state_batch_dims<S: ToTensor>(state_batch: &[&Rc<S>], tensor: &Tensor<f32>) {
        let declared_state_dims = || state_batch[0].dims();
        assert_eq!(tensor.dims().len(), 1 + declared_state_dims().len());
        // BATCH_SIZE
        assert_eq!(tensor.dims()[0], state_batch.len() as u64);
        assert_eq!(tensor.dims().split_first().unwrap().1, declared_state_dims());
    }
}

impl<S, A, const BATCH_SIZE: usize> QLearningModel<S, A, BATCH_SIZE> for QLearningTensorflowModel
where
    S: State + ToTensor,
    A: Action
{
    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [frame_size_x, frame_size_y, world_state_num_frames].
    ///   Representing the current state - this should be the last four frames of the required frame size
    ///   having one pixel encoded in a single float number
    ///
    fn predict_action(&self,
                      state: &Rc<S>,
    ) -> A {
        let state_tensor = state.to_tensor();
        assert_eq!(state_tensor.dims(), state.dims(), "state dimension mismatch from ToTensor::to_tensor(). Got {:?}, expected {:?}", state_tensor.dims(), state.dims());

        let r = self.fn_predict_single.apply::<_, i64>(&self.bundle.session, &state_tensor);
        let action = r[0] as ModelActionType;
        Action::try_from_numeric(action)
            .expect("action value should be in proper range")
    }

    fn batch_predict_future_reward(&self,
                                   state_batch: [&Rc<S>; BATCH_SIZE],
    ) -> [f32; BATCH_SIZE] {
        let state_batch_tensor = ToTensor::batch_to_tensor(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);

        let r: Tensor<f32> = self.fn_batch_predict_future_reward.apply(&self.bundle.session, &state_batch_tensor);
        assert_eq!(r.dims(), &[BATCH_SIZE as u64]);
        (*r).as_ref().try_into().unwrap()
    }

    /// Performs a single training step using a a batch of data.
    /// Returns the model's loss
    ///
    /// # Arguments
    /// * `state_batch` Tensor [BATCH_SIZE, frame_size_x, frame_size_y, world_state_num_frames]
    /// * `action_batch` Tensor [BATCH_SIZE, 1]
    /// * `updated_q_values` Tensor [BATCH_SIZE, 1]
    ///
    /// # Returns
    ///   calculated loss
    ///
    fn train(&self,
             state_batch: [&Rc<S>; BATCH_SIZE],
             action_batch: [A; BATCH_SIZE],
             updated_q_values: [f32; BATCH_SIZE],
    ) -> f32 {
        let state_batch_tensor = ToTensor::batch_to_tensor(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);

        let mut action_batch_tensor = Tensor::new(&[BATCH_SIZE as u64, 1]);
        for (i, action) in action_batch.into_iter().enumerate() {
            action_batch_tensor.set(&[i as u64, 0], action.numeric());
        }

        let mut updated_q_values_tensor = Tensor::new(&[BATCH_SIZE as u64, 1]);
        for (i, q) in updated_q_values.into_iter().enumerate() {
            updated_q_values_tensor.set(&[i as u64, 0], q);
        }
        let [r] = self.fn_train_model.apply::<_, _, _, f32>(
            &self.bundle.session,
            &state_batch_tensor,
            &action_batch_tensor,
            &updated_q_values_tensor);
        r[0]
    }

    fn write_checkpoint(&self,
                        file: &str,
    ) -> String {
        let r = self.fn_write_checkpoint.apply::<String, String>(
            &self.bundle.session,
            &Tensor::from(file.to_string()));
        r[0].clone()
    }

    fn read_checkpoint(&self,
                       file: &str,
    ) {
        self.fn_read_checkpoint.apply::<_, String>(
            &self.bundle.session,
            &Tensor::from(file.to_string()));
    }
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::rc::Rc;

    use rand::prelude::*;

    use crate::environment::breakout_environment::{BreakoutAction, BreakoutEnvironment};
    use crate::environment::util::frame_ring_buffer::FrameRingBuffer;
    use crate::ql::prelude::Action;

    use super::*;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tf_model/saved/q_learning_model_50x50x4to3")
    }

    const STATE_DIM: [u64; 3] = [50_u64, 50_u64, 4_u64];
    const ACTION_SPACE: ModelActionType = 3;
    const FRAME_SIZE_X: usize = 50;
    const FRAME_SIZE_Y: usize = 50;
    const BATCH_SIZE: usize = 32;

    #[test]
    fn test_load_model() {
        QLearningTensorflowModel::init(&model_dir());
    }

    #[test]
    fn test_predict_single() {
        let model = QLearningTensorflowModel::init(&model_dir());
        let state = Rc::new(FrameRingBuffer::new(FRAME_SIZE_X, FRAME_SIZE_Y));
        let action = model.predict_action(&state);
        log::info!("action: {}", action)
    }

    #[test]
    fn test_batch_predict_future_reward() {
        let model = QLearningTensorflowModel::init(&model_dir());
        let states = (0..BATCH_SIZE).map(|_| Rc::new(FrameRingBuffer::new(FRAME_SIZE_X, FRAME_SIZE_Y))).collect::<Vec<_>>();
        let state_batch: [&Rc<_>; BATCH_SIZE] = states.iter().collect::<Vec<_>>().try_into().unwrap();
        let _future_rewards = model.batch_predict_future_reward(state_batch);
    }

    #[test]
    fn test_train() {
        let model = QLearningTensorflowModel::init(&model_dir());
        let states = (0..BATCH_SIZE).map(|_| Rc::new(FrameRingBuffer::random(FRAME_SIZE_X, FRAME_SIZE_Y))).collect::<Vec<_>>();
        let state_batch: [&Rc<_>; BATCH_SIZE] = states.iter().collect::<Vec<_>>().try_into().unwrap();
        let action_batch = [0; BATCH_SIZE]
            .map(|_| thread_rng().gen_range(0..BreakoutAction::ACTION_SPACE))
            .map(|v| BreakoutAction::try_from_numeric(v).unwrap());
        let updated_q_values = [0; BATCH_SIZE].map(|_| thread_rng().gen_range(0.0..3.0));
        let loss = model.train(state_batch, action_batch, updated_q_values);
        log::info!("loss: {}", loss);
    }

    #[test]
    fn test_save_and_load_model_ckpt() {
        let keras_model_checkpoint_dir = tempfile::tempdir().unwrap();
        let keras_model_checkpoint_file = keras_model_checkpoint_dir.path().join("checkpoint");
        let model = QLearningTensorflowModel::init(&model_dir());

        let path = model.write_checkpoint(keras_model_checkpoint_file.to_str().unwrap());
        log::info!("saved model to '{}'", path);

        model.read_checkpoint(&keras_model_checkpoint_file.to_str().unwrap());
    }
}