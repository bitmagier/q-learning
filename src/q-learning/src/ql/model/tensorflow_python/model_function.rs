use anyhow::Result;
use tensorflow::{Graph, MetaGraphDef, Session, SessionRunArgs, Tensor, TensorType};

pub struct ModelFunction1<'a> {
    name: &'a str,
    p1_name: &'a str,
    out_name: &'a str,
}

impl<'a> ModelFunction1<'a> {
    pub fn new(
        name: &'a str,
        p1_name: &'a str,
        out_name: &'a str,
    ) -> Result<Self> {
        Ok(Self { name, p1_name, out_name })
    }

    pub fn apply<I1: TensorType, O: TensorType>(
        &self,
        graph: &Graph,
        meta_graph_def: &MetaGraphDef,
        session: &Session,
        p1: &Tensor<I1>,
    ) -> Result<Tensor<O>> {
        let signature = meta_graph_def.get_signature(self.name)?;
        let input1_operation = graph.operation_by_name_required(&signature.get_input(self.p1_name).unwrap().name().name)?;
        let output_operation = graph.operation_by_name_required(&signature.get_output(self.out_name).unwrap().name().name)?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&input1_operation, 0, p1);
        let out = args.request_fetch(&output_operation, 0);

        session.run(&mut args)?;
        Ok(args.fetch(out)?)
    }
}

pub struct ModelFunction3<'a> {
    name: &'a str,
    p1_name: &'a str,
    p2_name: &'a str,
    p3_name: &'a str,
    out_name: &'a str,
}

impl<'a> ModelFunction3<'a> {
    pub fn new(
        name: &'a str,
        p1_name: &'a str,
        p2_name: &'a str,
        p3_name: &'a str,
        out_name: &'a str,
    ) -> Result<Self> {
        Ok(Self {
            name,
            p1_name,
            p2_name,
            p3_name,
            out_name,
        })
    }

    pub fn apply<I1: TensorType, I2: TensorType, I3: TensorType, O: TensorType>(
        &self,
        graph: &Graph,
        meta_graph_def: &MetaGraphDef,
        session: &Session,
        arg1: &Tensor<I1>,
        arg2: &Tensor<I2>,
        arg3: &Tensor<I3>,
    ) -> Result<Tensor<O>> {
        let signature = meta_graph_def.get_signature(self.name)?;
        let input1_operation = graph.operation_by_name_required(&signature.get_input(self.p1_name).unwrap().name().name)?;
        let input2_operation = graph.operation_by_name_required(&signature.get_input(self.p2_name).unwrap().name().name)?;
        let input3_operation = graph.operation_by_name_required(&signature.get_input(self.p3_name).unwrap().name().name)?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&input1_operation, 0, arg1);
        args.add_feed(&input2_operation, 0, arg2);
        args.add_feed(&input3_operation, 0, arg3);

        let output_operation = graph.operation_by_name_required(&signature.get_output(self.out_name).unwrap().name().name)?;
        let out = args.request_fetch(&output_operation, 0);

        session.run(&mut args)?;
        Ok(args.fetch(out)?)
    }
}
