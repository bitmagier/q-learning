use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const KERAS_MODEL_DIR: &str = "python/keras_model/breakout_keras_model";

// const PRED_FUNCTION: TensorFunction = TensorFunction {
//     name: "pred",
//     input_param_names: vec!["inputs"],
//     output_param_name: "output_0",
// };

struct Tensors {
    graph: Graph,
    bundle: SavedModelBundle,
}

impl Tensors {
    pub fn new() -> Self {
        // Next we load the model as a graph from the path it was saved in
        let save_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(KERAS_MODEL_DIR);
        let mut graph = Graph::new();

        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, save_dir,
        ).expect("Can't load saved model");

        Tensors { graph, bundle }
    }

    // TODO implement a mini-batch training round
    fn train(&self) {
        let session = &self.bundle.session;
        let function_signature = self.bundle.meta_graph_def().get_signature("train").unwrap();
        /// These operations are representing nodes in a graph which represents a computation that produces an output.

        // Input information
        let training_input_param = function_signature.get_input("training_input").unwrap();
        let training_target_param = function_signature.get_input("training_target").unwrap();
        let output_param = function_signature.get_output("output_0").unwrap();

        let input_operation = self.graph.operation_by_name_required(&training_input_param.name().name).unwrap();
        let target_operation = self.graph.operation_by_name_required(&training_target_param.name().name).unwrap();
        let output_operation = self.graph.operation_by_name_required(&output_param.name().name).unwrap();

        // run the computation
        // The values will be fed to and retrieved from the model with this
        let mut args = SessionRunArgs::new();

        {
            // Feed the tensors into the graph
            let input_tensor: Tensor<f32> = Tensor::new(&[1, 2]).with_values(&[1.0, 1.0]).unwrap();
            let target_tensor: Tensor<f32> = Tensor::new(&[1, 1]).with_values(&[2.0]).unwrap();

            args.add_feed(&input_operation, 0, &input_tensor);
            args.add_feed(&target_operation, 0, &target_tensor);

            // Fetch result from graph
            let mut out = args.request_fetch(&output_operation, 0);

            session
                .run(&mut args)
                .expect("Error occurred during calculations");

            // The result will now be stored in the SessionRunArgs object. All that’s left is to retrieve it.
            // Here we take the value at index zero simply because there is only one value present.
            // in case of train this is the loss
            // (in case of pred it is a prediction value)
            let result: f32 = args.fetch(out).unwrap()[0];
        }



    }
}


