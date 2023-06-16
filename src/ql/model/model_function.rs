use tensorflow::{Graph, Operation, SavedModelBundle, Session, SessionRunArgs, Tensor, TensorType};

pub struct ModelFunction1 {
    name: String,
    input1_operation: Operation,
    output_operation: Operation,
}

impl ModelFunction1 {
    pub fn new(
        graph: &Graph,
        bundle: &SavedModelBundle,
        name: &str,
        input1_name: &str,
        output_name: &str,
    ) -> Self {
        let signature = bundle.meta_graph_def().get_signature(name).unwrap();
        Self {
            name: String::from(name),
            input1_operation: graph.operation_by_name_required(&signature.get_input(input1_name).unwrap().name().name).unwrap(),
            output_operation: graph.operation_by_name_required(&signature.get_output(output_name).unwrap().name().name).unwrap(),
        }
    }

    pub fn apply<I1: TensorType, O: TensorType>(
        &self,
        session: &Session,
        arg1: &Tensor<I1>,
    ) -> Tensor<O> {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input1_operation, 0, arg1);
        let out = args.request_fetch(&self.output_operation, 0);

        session.run(&mut args)
            .expect(&format!("error occurred while calling '{}'", self.name));
        args.fetch(out).unwrap()
    }
}

pub struct ModelFunction3 {
    name: String,
    input1_operation: Operation,
    input2_operation: Operation,
    input3_operation: Operation,
    output_operation: Operation,
}

impl ModelFunction3 {
    pub fn new(
        graph: &Graph,
        bundle: &SavedModelBundle,
        name: &str,
        input1_name: &str,
        input2_name: &str,
        input3_name: &str,
        output_name: &str,
    ) -> Self {
        let signature = bundle.meta_graph_def().get_signature(name).unwrap();
        Self {
            name: String::from(name),
            input1_operation: graph.operation_by_name_required(&signature.get_input(input1_name).unwrap().name().name).unwrap(),
            input2_operation: graph.operation_by_name_required(&signature.get_input(input2_name).unwrap().name().name).unwrap(),
            input3_operation: graph.operation_by_name_required(&signature.get_input(input3_name).unwrap().name().name).unwrap(),
            output_operation: graph.operation_by_name_required(&signature.get_output(output_name).unwrap().name().name).unwrap(),
        }
    }

    pub fn apply<I1: TensorType, I2: TensorType, I3: TensorType, O: TensorType>(
        &self,
        session: &Session,
        arg1: &Tensor<I1>,
        arg2: &Tensor<I2>,
        arg3: &Tensor<I3>,
    ) -> Tensor<O> {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input1_operation, 0, arg1);
        args.add_feed(&self.input2_operation, 0, arg2);
        args.add_feed(&self.input3_operation, 0, arg3);

        let out = args.request_fetch(&self.output_operation, 0);

        session.run(&mut args)
            .expect(&format!("error occurred while calling '{}'", self.name));
        args.fetch(out).unwrap()
    }
}
