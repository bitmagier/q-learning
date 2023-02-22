import tensorflow as tf
from keras.layers import Dense


class BreakoutModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(BreakoutModel, self).__init__(*args, **kwargs)
        self.dense_1 = Dense(2, name="test_in", input_dim=2)
        self.dense_2 = Dense(1, name="test_out")

    # Call function used to make predictions from Rust
    @tf.function
    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    # Train function called from Rust which uses the keras model innate train_step function
    @tf.function
    def training(self, train_data):
        loss = self.train_step(train_data)['loss']
        return loss


# Create model
my_model = BreakoutModel()
my_model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Get concrete function for the call and training method
pred_output = my_model.call.get_concrete_function(tf.TensorSpec(shape=[1, 2],
                                                                dtype=tf.float32,
                                                                name='inputs'))

train_output = my_model.training.get_concrete_function((tf.TensorSpec(shape=[1, 2],
                                                                      dtype=tf.float32, name="training_input"),
                                                        tf.TensorSpec(shape=[1, 1],
                                                                      dtype=tf.float32,
                                                                      name="training_target")))

# Save the model
my_model.save('breakout_keras_model',
              save_format='tf',
              signatures={'train': train_output, 'pred': pred_output})


# one way to get output names via saved_model_cli:
# saved_model_cli show --dir /path/to/saved-model/ --all
