#![allow(non_upper_case_globals)]

use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::Result;
use itertools::Itertools;
use lazy_static::lazy_static;
use tensorflow::{Graph, ImportGraphDefOptions, SavedModelBundle, SessionOptions, Tensor};

use crate::ql::model::tensorflow_python::model_function::{ModelFunction1, ModelFunction3};
use crate::ql::prelude::{Action, DeepQLearningModel, DEFAULT_BATCH_SIZE, Environment, ModelActionType, QlError, ToMultiDimArray};

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

        let fn_predict_single = ModelFunction1::new("predict_action", "state", "action")?;
        let fn_batch_predict_max_future_reward = ModelFunction1::new("batch_predict_max_future_reward", "state_batch", "reward_batch")?;
        let fn_train_model = ModelFunction3::new("train_model", "state_batch", "action_batch_one_hot", "updated_q_values", "loss")?;
        let fn_write_checkpoint = ModelFunction1::new("write_checkpoint", "file", "file")?;
        let fn_read_checkpoint = ModelFunction1::new("read_checkpoint", "file", "dummy")?;

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
    
    // pub fn save_model(&mut self, path: &Path) -> Result<()> {
    //     let mut b = SavedModelBuilder::new();
    //     b.add_tag("serve");
    //     
    //     for (name, sig) in self.bundle.meta_graph_def().signatures() {
    //         b.add_signature(name, sig.clone());
    //     }
    //     
    //     // TODO how to get variables?
    //     // b.add_collection()
    //     
    //     let saver = b.inject(&mut self.scope)?;
    //     saver.save(&self.bundle.session, &self.scope.graph(), path).map_err(|e| QlError(e.to_string()))?;
    //     Ok(())
    // }
    
    // save_model next try:
    // https://stackoverflow.com/questions/37508771/how-to-save-and-restore-a-tensorflow-graph-and-its-state-in-c
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
    ) -> Result<f32> {
        let state_batch_tensor = E::S::batch_to_multi_dim_array(&state_batch);
        Self::check_state_batch_dims(&state_batch, &state_batch_tensor);
        
        let mut action_batch_tensor = Tensor::<f32>::new(&[BATCH_SIZE as u64, E::A::ACTION_SPACE as u64]);
        for (i, action) in action_batch.into_iter().enumerate() {
            action_batch_tensor.set(&[i as u64, action.numeric() as u64], 1.0);
        }

        let r = self.fn_train_model.apply::<_, _, _, f32>(
            &self.graph,
            self.bundle.meta_graph_def(),
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
    ) -> Result<String> {
        let r = self
            .fn_write_checkpoint
            .apply::<String, String>(&self.graph, self.bundle.meta_graph_def(), &self.bundle.session, &Tensor::from(file.to_string()))?;
        Ok(r[0].clone())
    }

    fn read_checkpoint(
        &self,
        file: &str,
    ) -> Result<()> {
        self.fn_read_checkpoint
            .apply::<_, String>(&self.graph, self.bundle.meta_graph_def(), &self.bundle.session, &Tensor::from(file.to_string()))?;
        Ok(())
    }

    fn save_graph(
        &self,
        path: &Path,
    ) -> Result<()> {
        let serialized_graph = self.graph.graph_def()?;
        fs::write(path, &serialized_graph)?;
        Ok(())
    }

    fn load_graph(
         &mut self,
         path: &Path,
     ) -> Result<()> {
        let serialized_graph = fs::read(path)?;

        // TODO A: if we use the existing graph, we get Error: InvalidArgument: Node name 'Adam/action_layer/bias/v' already exists in the Graph
        // TODO B: if we use a fresh graph, we get a runtime error while using `batch_predict_max_future_reward` 
        //   (0) FAILED_PRECONDITION: Could not find variable dense/kernel. This could mean that the variable has been deleted. In TF1, it can also mean the variable is uninitialized. Debug info: container=localhost, status error message=Container localhost does not exist. (Could not find resource: localhost/dense/kernel)
        //     [[{{function_node __inference_batch_predict_max_future_reward_245}}{{node dense/MatMul/ReadVariableOp}}]]
        //     [[StatefulPartitionedCall/_3]]
        //
        // => Note: going back to try making use of SavedModelBuilder in order to save the model 
        
        // TODO prune graph before importing
        
        let import_options = ImportGraphDefOptions::new();
        // check if that helps and works
       
        let result = self.graph.import_graph_def_with_results(&serialized_graph, &import_options)?;
        
        let m = result.missing_unused_input_mappings()?;
        if !m.is_empty() {
            return Err(QlError(format!(
                "loaded graph does not seem to be consistent. We have missing unused mappings: {}",
                m.iter().map(|&(s, _)| s).join(", ")
            )))?;
        }
        
        // TODO initialize variables
        // HOW? 
        //   - maybe using a @tf.function calling  
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::ql::prelude::Action;
    use crate::test::ballgame_test_environment::{BallGameAction, BallGameTestEnvironment};

    use super::*;

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
        let loss = model.train(states_batch.each_ref(), action_batch, updated_q_values)?;
        log::info!("loss: {}", loss);
        Ok(())
    }

    // TODO add a test function for train, which trains an expected output for a very small set of input (maybe 2)


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

    // #[test]
    // fn test_save_and_load_model() -> Result<()> {
    //     let mut model = load_model()?;
    //     let temp_dir = tempfile::tempdir()?;
    //     let tf_graph_file = temp_dir.path().join("saved_model");
    //     model.save_graph(&tf_graph_file)?;
    //     log::info!("saved model to '{}'", &tf_graph_file.to_string_lossy());
    //     model.load_graph(&tf_graph_file)?;
    // 
    //     // make sure the temp directory is not dropped and removed before
    //     temp_dir.close()?;
    //     Ok(())
    // }
    
    // #[test]
    // fn test_save_and_load_via_saved_model_builder() -> Result<()>{
    //     let mut model = load_model()?;
    //     let tf_graph_file = Path::new("/tmp/tf").join("saved_model");
    //     model.save(&tf_graph_file)?;
    //     
    //     Ok(())
    // }
}

// One way to get output names via saved_model_cli:
// saved_model_cli show --dir /path/to/saved-model/ --all
