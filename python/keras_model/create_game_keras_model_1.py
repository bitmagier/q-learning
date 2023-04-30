import tensorflow as tf
from keras import layers, models


# example: https://www.tensorflow.org/tutorials/images/cnn
# another example: https://towardsdatascience.com/convolutional-neural-networks-with-tensorflow-2d0d41382d32

class GameModel1(models.Sequential):
    def __init__(self, *args, **kwargs):
        super(GameModel1, self).__init__(*args, **kwargs)
        # input: batches of normalized 2D Game screenshots (`tf.random.normal`)
        # effective input_shape: (batch-size, x, y, color-channels)

        # TODO dimensions
        # filters: integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        #          initial filters in a network is responsible for detecting edges and blobs; 16 or 32 is recommended
        # kernel_size
        #
        self.train_batch_size = 100
        self.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(600, 800, 3)))
        self.add(layers.MaxPooling2D((3, 3)))
        self.add(layers.Conv2D(32, (5, 5), activation='relu'))
        self.add(layers.MaxPooling2D((3, 3)))
        self.add(layers.Conv2D(64, (5, 5), activation='relu'))
        self.add(layers.MaxPooling2D((3, 3)))

        # TODO find optimal activation function candidates
        # relu, sigmoid, softmax, etc

        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dense(16, activation='sigmoid'))

        # number of neuron in the output layer matches the dimensionality of the action space (values range from 0..1)
        # < 0.4 means acceleration to left side
        # 0.4 .. 0.6 means stand still
        # > 0.6 means acceleration to right side
        self.add(layers.Dense(1, activation='sigmoid'))

        self.summary()

    @tf.function
    def x_call(self, inputs):
        output = self(inputs)
        return output

    @tf.function
    def x_training(self, train_data):
        loss = self.train_step(train_data)['loss']
        return loss


# (1)[https://www.tensorflow.org/guide/keras/sequential_model]
# (2)[https://keras.io/guides/sequential_model/#:~:text=A%20Sequential%20model%20is%20appropriate,tensor%20and%20one%20output%20tensor.&text=A%20Sequential%20model%20is%20not,multiple%20inputs%20or%20multiple%20outputs]
# (3)[https://pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/]

model = GameModel1()

# TODO optimization algorithms Stochastic Gradient Descent (SGD), RMSprop etc

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Get concrete function for the call and training method
# TODO add batch size as first parameter for `shape` (see conv2d.py)
pred_output = model.x_call.get_concrete_function(tf.TensorSpec(shape=[None, 600, 800, 3],
                                                               dtype=tf.float32,
                                                               name='inputs'))

# train_output = model.x_training.get_concrete_function((tf.TensorSpec(shape=[model.train_batch_size, 600, 800, 3],
#                                                                      dtype=tf.float32, name="training_input"),
#                                                        tf.TensorSpec(shape=[1],
#                                                                      dtype=tf.float32,
#                                                                      name="training_target")))

# Save the model
# model.save('game_keras_model_1',
#            save_format='tf',
#            signatures={'train': train_output, 'pred': pred_output})
model.save('game_keras_model_1',
           save_format='tf',
           signatures={'pred': pred_output})

# one way to get output names via saved_model_cli:
# saved_model_cli show --dir /path/to/saved-model/ --all


# TODO define a training mini-batch function
# e.g.
# - High level:
#   - input: last 5 images of the game
#   - output: action values: value vector for possible actions (left, none, right) [0..1]
# - steps:
#   - make a copy of the model
#   - run a min-batch (series of training data [input,target]
#           against the old model, but train the new one
#   - replace old model by the new model
#   - return stats (loss, etc)


# TODO were are the training parameter come from in the convolutional step?
