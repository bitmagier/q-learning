import tensorflow as tf
from keras.layers import Dense


class ExampleModel(tf.keras.Model):
    # TODO: Define 1 convolutional layer, 1 fully connected MLP
    def __init__(self, *args, **kwargs):
        super(ExampleModel, self).__init__(*args, **kwargs)
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

    # TODO define a training mini-batch function
    # - High level:
    #   - input: last 5 images of the game
    #   - output: action values: value vector for possible actions (left, none, right) [0..1]
    # - steps:
    #   - make a copy of the model
    #   - run a min-batch (series of training data [input,target]
    #           against the old model, but train the new one
    #   - replace old model by the new model
    #   - return stats (loss, etc)


# Create model
my_model = ExampleModel()
# optimizer = tf.keras.optimizers.experimental.SGD()
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
my_model.save('example_keras_model',
              save_format='tf',
              signatures={'train': train_output, 'pred': pred_output})

# one way to get output names via saved_model_cli:
# saved_model_cli show --dir /path/to/saved-model/ --all


# TODO create Network with convolutional layer + fully connected layer
# good example: https://www.tensorflow.org/tutorials/images/cnn
