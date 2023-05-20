#![allow(unused)]

use std::path::Path;

use tensorflow::{Graph, Operation, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor};

const KERAS_MODEL_DIR: &str = "keras_model/q_learning_model_1";

const BATCH_SIZE: u64 = 32;

pub struct QLearningModel {
    pub graph: Graph,
    pub bundle: SavedModelBundle,
    function_predict_single: FunctionPredictSingle,
    function_train_model: FunctionTrainModel,
}


impl QLearningModel {
    pub fn load() -> Self {
        // we load the model as a graph from the path it was saved in
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(KERAS_MODEL_DIR);
        let mut graph = Graph::new();

        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, model_dir,
        ).expect("Can't load model");

        // Prepare model function access for 'predict_single'
        //
        // One way to get output names via saved_model_cli:
        // saved_model_cli show --dir /path/to/saved-model/ --all

        let f_predict_single_signature = bundle.meta_graph_def().get_signature("predict_single").unwrap();
        let function_predict_single = FunctionPredictSingle {
            state_input_operation: graph.operation_by_name_required(&f_predict_single_signature.get_input("state").unwrap().name().name).unwrap(),
            output_operation: graph.operation_by_name_required(&f_predict_single_signature.get_output("action").unwrap().name().name).unwrap(),
        };

        // Prepare model function access for 'train_model'
        let f_train_model_signature = bundle.meta_graph_def().get_signature("train_model").unwrap();
        let function_train_model = FunctionTrainModel {
            state_samples_input_operation: graph.operation_by_name_required(&f_train_model_signature.get_input("state_samples").unwrap().name().name).unwrap(),
            action_samples_input_operation: graph.operation_by_name_required(&f_train_model_signature.get_input("action_samples").unwrap().name().name).unwrap(),
            updated_q_values_input_operation: graph.operation_by_name_required(&f_train_model_signature.get_input("updated_q_values").unwrap().name().name).unwrap(),
            output_operation: graph.operation_by_name_required(&f_train_model_signature.get_output("loss").unwrap().name().name).unwrap(),
        };

        QLearningModel {
            graph,
            bundle,
            function_predict_single,
            function_train_model,
        }
    }

    /// Predicts the next action based on the current state.
    ///
    /// # Arguments
    /// * `state` Game state Tensor [600, 800, 3]
    ///
    pub fn predict(
        &self,
        state: Tensor<f32>,
    ) -> i8 {
        self.function_predict_single.call(&self.bundle.session, state)
    }

    /// Performs a single training step using a a batch of data.
    /// Returns the model's loss
    ///
    /// # Arguments
    /// * `state_samples` Tensor [BATCH_SIZE, 600, 800, 3]
    /// * `action_samples` Tensor [BATCH_SIZE, 1]
    /// * `updated_q_values` Tensor [BATCH_SIZE, 1]
    ///
    pub fn train(
        &self,
        state_samples: Tensor<f32>,
        action_samples: Tensor<i8>,
        updated_q_values: Tensor<f32>,
    ) -> f32 {
        self.function_train_model.call(&self.bundle.session, state_samples, action_samples, updated_q_values)
    }
}


struct FunctionPredictSingle {
    state_input_operation: Operation,
    output_operation: Operation,
}

impl FunctionPredictSingle {
    pub fn call(
        &self,
        session: &Session,
        state: Tensor<f32>,
    ) -> i8 {
        // run the computation
        // The values will be fed to and retrieved from the model with this
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.state_input_operation, 0, &state);
        // Fetch result from graph
        let mut out = args.request_fetch(&self.output_operation, 0);

        session
            .run(&mut args)
            .expect("Error occurred during 'predict_single' calculations");

        // The result will now be stored in the SessionRunArgs object. All that’s left is to retrieve it.
        // Here we take the value at index zero simply because there is only one value present.
        // in case of train this is the loss
        // (in case of pred it is a prediction value)
        let result: i64 = args.fetch(out).unwrap()[0];
        result as i8
    }
}

struct FunctionTrainModel {
    state_samples_input_operation: Operation,
    action_samples_input_operation: Operation,
    updated_q_values_input_operation: Operation,
    output_operation: Operation,
}

impl FunctionTrainModel {
    pub fn call(
        &self,
        session: &Session,
        state_samples: Tensor<f32>,
        action_samples: Tensor<i8>,
        updated_q_values: Tensor<f32>,
    ) -> f32 {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.state_samples_input_operation, 0, &state_samples);
        args.add_feed(&self.action_samples_input_operation, 0, &action_samples);
        args.add_feed(&self.updated_q_values_input_operation, 0, &updated_q_values);

        let mut out = args.request_fetch(&self.output_operation, 0);

        session
            .run(&mut args)
            .expect("Error occurred during 'train_model' calculations");

        args.fetch(out).unwrap()[0]
    }
}

#[cfg(test)]
mod test {
    use tensorflow::Tensor;

    use crate::ai::q_learning_model_1::{BATCH_SIZE, QLearningModel};

    #[test]
    fn test_load_model() {
        QLearningModel::load();
    }

    #[test]
    fn test_predict() {
        let model = QLearningModel::load();
        let state = Tensor::new(&[600, 800, 3]);
        model.predict(state);
    }

    #[test]
    fn test_train() {
        let model = QLearningModel::load();
        let state_samples = Tensor::new(&[BATCH_SIZE, 600, 800, 3]);
        let action_samples = Tensor::new(&[BATCH_SIZE, 1]);
        let updated_q_values = Tensor::new(&[BATCH_SIZE, 1]);

        let _loss = model.train(state_samples, action_samples, updated_q_values);
    }
}
