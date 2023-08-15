Works:
- Breakout egui user-driven game: 
  `cd src && cargo run -p breakout-game --release`
- Defining and using a model in Python with Tensorflow/Keras (see `src/q-learning/python_model`)
- Using a SavedModel (defined and saved in Python) in Rust for inference and learning. (Unfortunately we can not restore saved model weights directly using Rust). 
  See integration tests in `src/q-learning`

Next:
- Doing the q-learning in Python using a Rust library to feed it with the agent environment data (State, Action, Reward)
  (Pyo3 seems to be the way to call Rust from Python)