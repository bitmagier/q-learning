# Setup
In order to do something useful with that code here, one needs to install or self-compile tensorflow first.
You might find details in [tensorflow_install.md](doc/tensorflow_install.md) useful.

Second essential thing to do is creating a saved keras model using `python` and the appropriate python script in `tf_model`.
E.g. for the integration-test we need:
```
cd tf_model && python create_ql_model_ballgame_5x5x3_4_256.py
```
BTW: You will see here too, whether your tensorflow installation works or not.

