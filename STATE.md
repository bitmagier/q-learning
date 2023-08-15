Works:
- Breakout egui user-driven game
- Defining a model in Python with Tensorflow/Keras 
- Using a SavedModel (defined and saved in Python) in Rust for inference and learning. (Unfortunately we can not restore saved model weights directly using Rust)

Next:
- Do the q-learning in Python using a Rust library to feed it with the agent environment data (State, Action, Reward)
  (Pyo3 seems to be the way to call Rust from Python)