#![allow(non_upper_case_globals)]

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::Result;
use lazy_static::lazy_static;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, Tensor};

use crate::ql::model::tensorflow::model_function::{ModelFunction1, ModelFunction3};
use crate::ql::prelude::{Action, DEFAULT_BATCH_SIZE, Environment, ModelActionType, DeepQLearningModel, ToMultiDimArray};

lazy_static!(
    pub static ref QL_MODEL_BALLGAME_3x3x4_5_512_PATH: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tf_model/saved/ql_model_ballgame_3x3x4_5_512");
);

pub struct QLearningTensorflowModel<E, const BATCH_SIZE: usize = DEFAULT_BATCH_SIZE> {
    bundle: SavedModelBundle,
    fn_predict_single: ModelFunction1,
    fn_batch_predict_max_future_reward: ModelFunction1,
    fn_train_model: ModelFunction3,
    fn_write_checkpoint: ModelFunction1,
    fn_read_checkpoint: ModelFunction1,
    _phantom: PhantomData<E>,
}

impl<E, const BATCH_SIZE: usize> QLearningTensorflowModel<E, BATCH_SIZE>
where
    E: Environment,
    <E as Environment>::S: ToMultiDimArray<Tensor<f32>>,
{
    /// Init / Load model
    ///
    /// # Arguments
    /// * `model_dir` saved tensorflow model;  e.g. Path::new(env!("CARGO_MANIFEST_DIR")).join("tf_model/saved/q_learning_model_50x50x4to3")
    /// * `model_state_dim` input dimensions of the model; e.g. [600x600x4] describing four 600x600 grayscale frames; (used for input data versus model dim reference checks)
    /// * `action_space` number of possible actions = size of the model's one dimension output vector; (used for input data versus model dim reference checks)
    pub fn load(model_dir: &Path) -> Self {
        // we load the model as a graph from the path it was saved in
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            ["serve"],
            &mut graph,
            model_dir.to_str().expect("should have UTF-8 compatible path"),
        )
            .expect("Can't load model");

        // One way to get output names via saved_model_cli:
        // saved_model_cli show --dir /path/to/saved-model/ --all

        let fn_predict_single = ModelFunction1::new(&graph, &bundle, "predict_action", "state", "action");
        let fn_batch_predict_max_future_reward =
            ModelFunction1::new(&graph, &bundle, "batch_predict_max_future_reward", "state_batch", "reward_batch");
        let fn_train_model = ModelFunction3::new(
            &graph,
            &bundle,
            "train_model",
            "state_batch",
            "action_batch_one_hot",
            "updated_q_values",
            ["loss"],
        );
        let fn_write_checkpoint = ModelFunction1::new(&graph, &bundle, "write_checkpoint", "file", "file");
        let fn_read_checkpoint = ModelFunction1::new(&graph, &bundle, "read_checkpoint", "file", "dummy");

        QLearningTensorflowModel {
            bundle,
            fn_predict_single,
            fn_batch_predict_max_future_reward,
            fn_train_model,
            fn_write_checkpoint,
            fn_read_checkpoint,
            _phantom: PhantomData,
        }
    }

    fn check_state_batch_dims(
        state_batch: &[&Rc<E::S>],
        tensor: &Tensor<f32>,
    ) {
        let declared_state_dims = || state_batch[0].dims();
        assert_eq!(tensor.dims().len(), 1 + declared_state_dims().len());
        // check BATCH_SIZE
        assert_eq!(tensor.dims()[0], state_batch.len() as u64);
        assert_eq!(tensor.dims().split_first().unwrap().1, declared_state_dims());
    }
}

impl<E, const BATCH_SIZE: usize> DeepQLearningModel<BATCH_SIZE> for QLearningTensorflowModel<E, BATCH_SIZE>
where
    E: Environment,
    <E as Environment>::S: ToMultiDimArray<Tensor<f32>>,
{
    type E = E;

    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [frame_size_x, frame_size_y, world_state_num_frames].
    ///   Representing the current state - this should be the last four frames of the required frame size
    ///   having one pixel encoded in a single float number
    ///
    fn predict_action(
        &self,
        state: &E::S,
    ) -> E::A {
        let state_tensor = state.to_multi_dim_array();
        assert_eq!(
            state_tensor.dims(),
            state.dims(),
            "state dimension mismatch from ToTensor::to_tensor(). Got {:?}, expected {:?}",
            state_tensor.dims(),
            state.dims()
        );

        let r = self.fn_predict_single.apply::<_, i64>(&self.bundle.session, &state_tensor);
        log::trace!("predict_action result: {:?}", r);

        let action = r[0] as ModelActionType;
        Action::try_from_numeric(action).expect("action value should be in proper range")
    }

    fn batch_predict_max_future_reward(
        &self,
        state_batch: [&Rc<E::S>; BATCH_SIZE],
    ) -> [f32; BATCH_SIZE] {
        let state_batch_tensor: Tensor<f32> = E::S::batch_to_multi_dim_array(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);

        let r: Tensor<f32> = self
            .fn_batch_predict_max_future_reward
            .apply(&self.bundle.session, &state_batch_tensor);
        assert_eq!(r.dims(), &[BATCH_SIZE as u64]);
        log::trace!("batch_predict result: {:?}", r);
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
    fn train(
        &self,
        state_batch: [&Rc<E::S>; BATCH_SIZE],
        action_batch: [E::A; BATCH_SIZE],
        updated_q_values: [f32; BATCH_SIZE],
    ) -> Result<f32> {
        let state_batch_tensor = E::S::batch_to_multi_dim_array(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);

        let mut action_batch_tensor = Tensor::<f32>::new(&[BATCH_SIZE as u64, E::A::ACTION_SPACE as u64]);
        for (i, action) in action_batch.into_iter().enumerate() {
            action_batch_tensor.set(&[i as u64, action.numeric() as u64], 1.0);
        }

        let [r] = self.fn_train_model.apply::<_, _, _, f32>(
            &self.bundle.session,
            &state_batch_tensor,
            &action_batch_tensor,
            &Tensor::from(updated_q_values),
        )?;
        Ok(r[0])
    }

    fn write_checkpoint(
        &self,
        file: &str,
    ) -> String {
        let r = self
            .fn_write_checkpoint
            .apply::<String, String>(&self.bundle.session, &Tensor::from(file.to_string()));
        r[0].clone()
    }

    fn read_checkpoint(
        &self,
        file: &str,
    ) {
        self.fn_read_checkpoint
            .apply::<_, String>(&self.bundle.session, &Tensor::from(file.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use crate::environment::ballgame_test_environment::{BallGameAction, BallGameTestEnvironment};

    use crate::ql::prelude::Action;

    use super::*;

    const BATCH_SIZE: usize = 512;

    fn load_model() -> QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE> {
        QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH)
    }

    #[test]
    fn test_load_model() { load_model(); }

    #[test]
    fn test_predict_single() {
        let model = load_model();
        let env = BallGameTestEnvironment::default();
        let action: BallGameAction = model.predict_action(&env.state());
        log::info!("action: {}", action)
    }

    #[test]
    fn test_batch_predict_max_future_reward() {
        let model = load_model();
        let mut env = BallGameTestEnvironment::default();
        let states_batch = [0; BATCH_SIZE].map(|_| {
            for _ in 0..5 {
                let action = BallGameAction::try_from_numeric(thread_rng().gen_range(0..BallGameAction::ACTION_SPACE)).unwrap();
                env.step(action);
            }
            Rc::new(env.state().clone())
        });
        let _future_rewards = model.batch_predict_max_future_reward(states_batch.each_ref());
    }

    #[test]
    fn test_train_function_call() -> Result<()> {
        let model = load_model();
        let env = BallGameTestEnvironment::default();
        let states_batch = [0; BATCH_SIZE].map(|_| Rc::new(env.state().clone()));
        let action_batch = [0; BATCH_SIZE]
            .map(|_| thread_rng().gen_range(0..BallGameAction::ACTION_SPACE))
            .map(|v| BallGameAction::try_from_numeric(v).unwrap());
        let updated_q_values = [0; BATCH_SIZE].map(|_| thread_rng().gen_range(0.0..1.5));
        let loss = model.train(states_batch.each_ref(), action_batch, updated_q_values)?;
        log::info!("loss: {}", loss);
        Ok(())
    }

    // TODO add a test function for train, which trains an expected output for a very small set of input (maybe 2)

    #[test]
    fn test_save_and_load_model_ckpt() {
        let keras_model_checkpoint_dir = tempfile::tempdir().unwrap();
        let keras_model_checkpoint_file = keras_model_checkpoint_dir.path().join("checkpoint");
        let model = load_model();
        let path = model.write_checkpoint(keras_model_checkpoint_file.to_str().unwrap());
        log::info!("saved model to '{}'", path);
        model.read_checkpoint(&keras_model_checkpoint_file.to_str().unwrap());
    }
}
