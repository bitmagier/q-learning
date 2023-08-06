// use std::env;
// use std::error::Error;
// use std::fs;
// use std::io::ErrorKind;
// use std::path::Path;
// use std::result::Result;
// use tensorflow::AttrType::Placeholder;
// use tensorflow::ops;
// use tensorflow::train::AdadeltaOptimizer;
// use tensorflow::train::MinimizeOptions;
// use tensorflow::train::Optimizer;
// use tensorflow::Code;
// use tensorflow::DataType;
// use tensorflow::Graph;
// use tensorflow::Output;
// use tensorflow::OutputName;
// use tensorflow::SavedModelBundle;
// use tensorflow::Scope;
// use tensorflow::Session;
// use tensorflow::SessionOptions;
// use tensorflow::SessionRunArgs;
// use tensorflow::Shape;
// use tensorflow::SignatureDef;
// use tensorflow::Status;
// use tensorflow::Tensor;
// use tensorflow::TensorInfo;
// use tensorflow::Variable;
// use tensorflow::REGRESS_INPUTS;
// use tensorflow::REGRESS_METHOD_NAME;
// use tensorflow::REGRESS_OUTPUTS;
// 
// // Helper for building a layer.
// //
// // `activation` is a function which takes a tensor and applies an activation
// // function such as tanh.
// //
// // Returns variables created and the layer output.
// fn layer<O1: Into<Output>>(
//     input: O1,
//     input_size: u64,
//     output_size: u64,
//     activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
//     scope: &mut Scope,
// ) -> Result<(Vec<Variable>, Output), Status> {
//     let mut scope = scope.new_sub_scope("layer");
//     let scope = &mut scope;
//     let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
//     let w = Variable::builder()
//         .initial_value(
//             ops::RandomStandardNormal::new()
//                 .dtype(DataType::Float)
//                 .build(w_shape, scope)?,
//         )
//         .data_type(DataType::Float)
//         .shape([input_size, output_size])
//         .build(&mut scope.with_op_name("w"))?;
//     let b = Variable::builder()
//         .const_initial_value(Tensor::<f32>::new(&[output_size]))
//         .build(&mut scope.with_op_name("b"))?;
//     Ok((
//         vec![w.clone(), b.clone()],
//         activation(
//             ops::add(
//                 ops::mat_mul(input, w.output().clone(), scope)?,
//                 b.output().clone(),
//                 scope,
//             )?
//                 .into(),
//             scope,
//         )?,
//     ))
// }
// 
// fn train<P: AsRef<Path>>(save_dir: P) -> Result<(), Box<dyn Error>> {
//     // ================
//     // Build the model.
//     // ================
//     let mut scope = Scope::new_root_scope();
//     let scope = &mut scope;
//     // Size of the hidden layer.
//     // This is far more than is necessary, but makes it train more reliably.
//     let hidden_size: u64 = 8;
//     let input = Placeholder::new()
//         .dtype(DataType::Float)
//         .shape([1u64, 2])
//         .build(&mut scope.with_op_name("input"))?;
//     let label = Placeholder::new()
//         .dtype(DataType::Float)
//         .shape([1u64])
//         .build(&mut scope.with_op_name("label"))?;
//     // Hidden layer.
//     let (vars1, layer1) = layer(
//         input.clone(),
//         2,
//         hidden_size,
//         &|x, scope| Ok(ops::tanh(x, scope)?.into()),
//         scope,
//     )?;
//     // Output layer.
//     let (vars2, layer2) = layer(layer1.clone(), hidden_size, 1, &|x, _| Ok(x), scope)?;
//     let error = ops::sub(layer2.clone(), label.clone(), scope)?;
//     let error_squared = ops::mul(error.clone(), error, scope)?;
//     let mut optimizer = AdadeltaOptimizer::new();
//     optimizer.set_learning_rate(ops::constant(1.0f32, scope)?);
//     let mut variables = Vec::new();
//     variables.extend(vars1);
//     variables.extend(vars2);
//     let (minimizer_vars, minimize) = optimizer.minimize(
//         scope,
//         error_squared.clone().into(),
//         MinimizeOptions::default().with_variables(&variables),
//     )?;
// 
//     let mut all_vars = variables.clone();
//     all_vars.extend_from_slice(&minimizer_vars);
//     let mut builder = tensorflow::SavedModelBuilder::new();
//     builder
//         .add_collection("train", &all_vars)
//         .add_tag("serve")
//         .add_tag("train")
//         .add_signature(REGRESS_METHOD_NAME, {
//             let mut def = SignatureDef::new(REGRESS_METHOD_NAME.to_string());
//             def.add_input_info(
//                 REGRESS_INPUTS.to_string(),
//                 TensorInfo::new(
//                     DataType::Float,
//                     Shape::from(None),
//                     OutputName {
//                         name: input.name()?,
//                         index: 0,
//                     },
//                 ),
//             );
//             def.add_output_info(
//                 REGRESS_OUTPUTS.to_string(),
//                 TensorInfo::new(DataType::Float, Shape::from(None), layer2.name()?),
//             );
//             def
//         });
//     let saved_model_saver = builder.inject(scope)?;
// 
//     // =========================
//     // Initialize the variables.
//     // =========================
//     let options = SessionOptions::new();
//     let g = scope.graph_mut();
//     let session = Session::new(&options, &g)?;
//     let mut run_args = SessionRunArgs::new();
//     // Initialize variables we defined.
//     for var in &variables {
//         run_args.add_target(&var.initializer());
//     }
//     // Initialize variables the optimizer defined.
//     for var in &minimizer_vars {
//         run_args.add_target(&var.initializer());
//     }
//     session.run(&mut run_args)?;
// 
//     // ================
//     // Train the model.
//     // ================
//     let mut input_tensor = Tensor::<f32>::new(&[1, 2]);
//     let mut label_tensor = Tensor::<f32>::new(&[1]);
//     // Helper that generates a training example from an integer, trains on that
//     // example, and returns the error.
//     let mut train = |i| -> Result<f32, Box<dyn Error>> {
//         input_tensor[0] = (i & 1) as f32;
//         input_tensor[1] = ((i >> 1) & 1) as f32;
//         label_tensor[0] = ((i & 1) ^ ((i >> 1) & 1)) as f32;
//         let mut run_args = SessionRunArgs::new();
//         run_args.add_target(&minimize);
//         let error_squared_fetch = run_args.request_fetch(&error_squared, 0);
//         run_args.add_feed(&input, 0, &input_tensor);
//         run_args.add_feed(&label, 0, &label_tensor);
//         session.run(&mut run_args)?;
//         Ok(run_args.fetch::<f32>(error_squared_fetch)?[0])
//     };
//     for i in 0..10000 {
//         train(i)?;
//     }
// 
//     // ================
//     // Save the model.
//     // ================
//     saved_model_saver.save(&session, &g, &save_dir)?;
// 
//     // ===================
//     // Evaluate the model.
//     // ===================
//     for i in 0..4 {
//         let error = train(i)?;
//         println!("Error: {}", error);
//         if error > 0.1 {
//             return Err(Box::new(Status::new_set(
//                 Code::Internal,
//                 &format!("Error too high: {}", error),
//             )?));
//         }
//     }
//     Ok(())
// }