## What already works

- Breakout egui user-driven game:

```shell 
cd src && cargo run -p breakout-game --release
```

- Defining and using a model in Python with Tensorflow/Keras (see `src/q-learning/python_model`)
- Using a SavedModel (defined and saved in Python) in Rust for inference and learning.
  __We are able to train our integration test model__ `ql_model_ballgame_3x3x4_5_512` in Rust so that it is able to solve the ballgame task.    
  => Unfortunately I discovered, that its apparently not yet supported by Tensorflow to restore the saved model weights directly using Rust code.
  See `readme.md` and integration tests in `src/ql-in-rust`

- Integration test:

```sh
git clone git@github.com:bitmagier/q-learning-breakout.git
cd q-learning-breakout
cd tf_model
python3 create_ql_model_ballgame_3x3x4_5_512.py
cd ..

cargo test --test learn_ballgame --release
```

## What's next

- Doing the Q-learning in Python using a Rust library to feed it with the agent environment data (State, Action, Reward)
  (Pyo3 seems to be the way to call Rust from Python)
