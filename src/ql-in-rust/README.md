# Setup
In order to do something useful with that code here, one needs to install or self-compile tensorflow first.
Also a python installation is required.
You might find details in [tensorflow_install.md](doc/tensorflow_install.md) useful.


This is the procedure for running the integration-tests:

0. First we expect to have no SavedModel present under `python_model/saved` and no checkpoint under `python_model/checkpoints`
1. Create the initial Keras SavedModel (having random weights)
2. Run rust q-learning integration test "learn_ballgame" - producing a checkpoint (may take ~15 minutes, depends on your hardware)
3. Create the SavedModel again, now including the learned weights from the written checkpoint
4. Run rust integration test "render_ballgame_cases"  
```sh
cd python_model
python create_ql_model_ballgame_3x3x4_5_512.py
cargo test --test learn_ballgame --release
python create_ql_model_ballgame_3x3x4_5_512.py
cargo test --test render_ballgame_cases
```

