#![allow(unused)]

use std::path::Path;

use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const KERAS_MODEL_DIR: &str = "keras_model/q_learning_model_1";

const BATCH_SIZE: i32 = 32;

struct FunctionPredictSingle {
    state_input_operation: Operation,
    output_operation: Operation,
}

struct FunctionTrainModel {
    state_samples_input_operation: Operation,
    action_samples_input_operation: Operation,
    updated_q_values_input_operation: Operation,
    output_operation: Operation,
}

struct QLearningModel {
    graph: Graph,
    bundle: SavedModelBundle,
    function_predict_single: FunctionPredictSingle,
    function_train_model: FunctionTrainModel
}

impl QLearningModel {
    pub fn new() -> Self {
        // Next we load the model as a graph from the path it was saved in
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(KERAS_MODEL_DIR);
        let mut graph = Graph::new();

        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, model_dir,
        ).expect("Can't load model");


        // Prepare model function access for 'predict_single'
        let f_predict_single_signature = bundle.meta_graph_def().get_signature("predict_single").unwrap();
        let function_predict_single = FunctionPredictSingle {
            state_input_operation: graph.operation_by_name_required(&f_predict_single_signature.get_input("state").unwrap().name().name).unwrap(),
            output_operation: graph.operation_by_name_required(&f_predict_single_signature.get_output("action").unwrap().name().name).unwrap(),
        };

        // Prepare model function access for 'train_model'
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
            function_train_model
        }
    }

    /// Calls the same named tensorflow model function
    ///
    /// # Arguments
    /// * `state` Game state Tensor [600, 800, 3]
    ///
    //TODO move to FunctionPredictSingle
    fn predict_single(
        &self,
        state: Tensor<f32>,
    ) -> i8 {
        let session = &self.bundle.session;

        // run the computation
        // The values will be fed to and retrieved from the model with this
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.function_predict_single.state_input_operation, 0, &state);
        // Fetch result from graph
        let mut out = args.request_fetch(&self.function_predict_single.output_operation, 0);

        session
            .run(&mut args)
            .expect("Error occurred during 'predict_single' calculations");

        // The result will now be stored in the SessionRunArgs object. All that’s left is to retrieve it.
        // Here we take the value at index zero simply because there is only one value present.
        // in case of train this is the loss
        // (in case of pred it is a prediction value)
        let result = args.fetch(out).unwrap()[0];
        result as i8
    }

    /// # Arguments
    /// * `state_samples` Tensor [BATCH_SIZE, 600, 800, 3]
    /// * `action_samples` Tensor [BATCH_SIZE, 1]
    /// * `updated_q_values` Tensor [BATCH_SIZE, 1]
    fn train_model(&self, state_samples: Tensor<f32>, action_samples: Tensor<i8>, updated_q_values: Tensor<f32>) -> f32 {
        let session = &self.bundle.session;
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.f_train_model_input_operation, 0, &state_samples);
        args.add_feed(&self.f_train_model_input_operation, 0, &action_samples);
        args.add_feed(&self.f_train_model_input_operation, 0, &updated_q_values);

        let mut out = args.request_fetch(&f_train_model_output_operation, 0);

        session
            .run(&mut args)
            .expect("Error occured during 'train_model' calculations");

        args.fetch(out).unwrap()[0]
    }
}


