#![allow(non_upper_case_globals)]

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::Result;
use itertools::Itertools;
use lazy_static::lazy_static;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, Tensor};

use crate::ql::ml_model::model::{DeepQLearningModel, ToMultiDimArray, DEFAULT_BATCH_SIZE};
use crate::ql::ml_model::tensorflow_python::model_function::{ModelFunction1, ModelFunction3};
use crate::ql::prelude::{Action, Environment, ModelActionType};

lazy_static! {
    pub static ref QL_MODEL_BALLGAME_3x3x4_5_512_PATH: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_model/saved/ql_model_ballgame_3x3x4_5_512");
}

pub struct QLearningTensorflowModel<E, const BATCH_SIZE: usize = DEFAULT_BATCH_SIZE> {
    graph: Graph,
    bundle: SavedModelBundle,
    fn_predict_single: ModelFunction1<'static>,
    fn_batch_predict_max_future_reward: ModelFunction1<'static>,
    fn_train_model: ModelFunction3<'static>,
    fn_write_checkpoint: ModelFunction1<'static>,
    fn_read_checkpoint: ModelFunction1<'static>,
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
    pub fn load_model(model_dir: &Path) -> Result<Self> {
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            ["serve"],
            &mut graph,
            model_dir.to_str().expect("should have UTF-8 compatible path"),
        )?;

        log::debug!(
            "available operations: {}",
            graph.operation_iter().map(|o| o.name().unwrap()).join(",")
        );
        log::debug!(
            "functions: {}",
            graph.get_functions()?.iter().map(|f| f.get_name().unwrap()).join(" ,")
        );

        let fn_predict_single = ModelFunction1::new("predict_action", "state", "action")?;
        let fn_batch_predict_max_future_reward = ModelFunction1::new("batch_predict_max_future_reward", "state_batch", "reward_batch")?;
        let fn_train_model = ModelFunction3::new("train_model", "state_batch", "action_batch_one_hot", "updated_q_values", "i")?;
        let fn_write_checkpoint = ModelFunction1::new("write_checkpoint", "file", "file")?;
        let fn_read_checkpoint = ModelFunction1::new("read_checkpoint", "file", "status")?;

        Ok(QLearningTensorflowModel {
            graph,
            bundle,
            fn_predict_single,
            fn_batch_predict_max_future_reward,
            fn_train_model,
            fn_write_checkpoint,
            fn_read_checkpoint,
            _phantom: PhantomData,
        })
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

        let r = match self
            .fn_predict_single
            .apply::<_, i64>(&self.graph, self.bundle.meta_graph_def(), &self.bundle.session, &state_tensor)
        {
            Ok(r) => r,
            Err(err) => panic!("{}", err),
        };
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

        let r: Tensor<f32> = match self.fn_batch_predict_max_future_reward.apply(
            &self.graph,
            self.bundle.meta_graph_def(),
            &self.bundle.session,
            &state_batch_tensor,
        ) {
            Ok(r) => r,
            Err(err) => panic!("{}", err),
        };
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
    ) -> Result<()> {
        let state_batch_tensor = E::S::batch_to_multi_dim_array(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);

        let mut action_batch_tensor = Tensor::<f32>::new(&[BATCH_SIZE as u64, E::A::ACTION_SPACE as u64]);
        for (i, action) in action_batch.into_iter().enumerate() {
            action_batch_tensor.set(&[i as u64, action.numeric() as u64], 1.0);
        }

        let _ = self.fn_train_model.apply::<_, _, _, i64>(
            &self.graph,
            self.bundle.meta_graph_def(),
            &self.bundle.session,
            &state_batch_tensor,
            &action_batch_tensor,
            &Tensor::from(updated_q_values),
        )?;
        // r[0] = returned 'number of iterations' (to return something from the graph, which is required)
        Ok(())
    }

    fn write_checkpoint(
        &self,
        file: &str,
    ) -> Result<String> {
        let r = self.fn_write_checkpoint.apply::<String, String>(
            &self.graph,
            self.bundle.meta_graph_def(),
            &self.bundle.session,
            &Tensor::from(file.to_string()),
        )?;
        Ok(r[0].clone())
    }

    /// Beware: Restoring a checkpoint in a Tensorflow graph function does not seem to have any impact to the weights of the model
    fn read_checkpoint(
        &self,
        file: &str,
    ) -> Result<()> {
        let _ = self.fn_read_checkpoint.apply::<_, i32>(
            &self.graph,
            self.bundle.meta_graph_def(),
            &self.bundle.session,
            &Tensor::from(file.to_string()),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use super::*;
    use crate::ql::prelude::Action;
    use crate::test::ballgame_test_environment::{BallGameAction, BallGameTestEnvironment};

    const BATCH_SIZE: usize = 512;

    fn load_model() -> Result<QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>> {
        QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load_model(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH)
    }

    #[test]
    fn test_load_model() -> Result<()> {
        load_model()?;
        Ok(())
    }

    #[test]
    fn test_predict_single() -> Result<()> {
        let model = load_model()?;
        let env = BallGameTestEnvironment::default();
        let action: BallGameAction = model.predict_action(&env.state());
        log::info!("action: {}", action);
        Ok(())
    }

    #[test]
    fn test_batch_predict_max_future_reward() -> Result<()> {
        let model = load_model()?;
        let mut env = BallGameTestEnvironment::default();
        let states_batch = [0; BATCH_SIZE].map(|_| {
            for _ in 0..5 {
                let action = BallGameAction::try_from_numeric(thread_rng().gen_range(0..BallGameAction::ACTION_SPACE)).unwrap();
                env.step(action);
            }
            Rc::new(env.state().clone())
        });
        let _future_rewards = model.batch_predict_max_future_reward(states_batch.each_ref());
        Ok(())
    }

    #[test]
    fn test_train_function_call() -> Result<()> {
        let model = load_model()?;
        let env = BallGameTestEnvironment::default();
        let states_batch = [0; BATCH_SIZE].map(|_| Rc::new(env.state().clone()));
        let action_batch = [0; BATCH_SIZE]
            .map(|_| thread_rng().gen_range(0..BallGameAction::ACTION_SPACE))
            .map(|v| BallGameAction::try_from_numeric(v).unwrap());
        let updated_q_values = [0; BATCH_SIZE].map(|_| thread_rng().gen_range(0.0..1.5));
        model.train(states_batch.each_ref(), action_batch, updated_q_values)?;
        Ok(())
    }

    // TODO check that we really restored what we saved before! (but how?)
    #[test]
    fn test_save_and_load_model_ckpt() -> Result<()> {
        let keras_model_checkpoint_dir = tempfile::tempdir()?;
        let keras_model_checkpoint_file = keras_model_checkpoint_dir.path().join("checkpoint");
        let model = load_model()?;
        let path = model.write_checkpoint(keras_model_checkpoint_file.to_str().unwrap())?;
        log::info!("saved model to '{}'", path);
        model.read_checkpoint(&keras_model_checkpoint_file.to_str().unwrap())?;
        Ok(())
    }
}

// One way to get output names via saved_model_cli:
// saved_model_cli show --dir /path/to/saved-model/ --all
