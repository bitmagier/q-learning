import tensorflow as tf
from keras import layers, optimizers

# These are the top references to follow while finding a suitable implementation:
# https://keras.io/examples/rl/deep_q_network_breakout/
# https://github.com/tensorflow/rust/tree/master/examples

WORLD_STATE_FRAMES = 4
ACTION_SPACE = 3
BATCH_SIZE = 32  # Size of batch taken from replay buffer


class QLearningModel(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super(QLearningModel, self).__init__(*args, **kwargs)
        self.add(layers.Conv2D(32, 12, strides=6, activation='relu', input_shape=(600, 800, WORLD_STATE_FRAMES),
                               name='convolution_layer1'))
        self.add(layers.Conv2D(64, 6, strides=3, activation='relu', name='convolution_layer2'))
        self.add(layers.Conv2D(64, 4, strides=2, activation='relu', name='convolution_layer3'))
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(512, activation='relu', name='full_layer'))
        self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))

        self.compile(optimizer=optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
                     # Using huber loss for stability
                     loss=tf.keras.losses.Huber(),
                     metrics=['accurate'])

    # Predict action from environment state
    @tf.function(input_signature=[tf.TensorSpec(shape=[600, 800, WORLD_STATE_FRAMES], dtype=tf.float32, name='state')])
    def predict_single(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0])
        return {'action': action}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, 600, 800, WORLD_STATE_FRAMES], dtype=tf.float32,
                      name='state_samples'),
        tf.TensorSpec(shape=[BATCH_SIZE, 1], dtype=tf.int8, name='action_samples'),
        tf.TensorSpec(shape=[BATCH_SIZE, 1], dtype=tf.float32, name='updated_q_values')
    ])
    def train_model(self, state_samples, action_samples, updated_q_values):
        # Create a mask - so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_samples, ACTION_SPACE)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self(state_samples)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='path')])
    def write_checkpoint(self, path):
        checkpoint = tf.train.Checkpoint(self)
        path = checkpoint.write(path)
        return {'path': tf.convert_to_tensor(path)}

    # @tf.function(input_signature=[])
    # def get_weights(self):
    #     return tf.keras.Sequential.get_weights(self)
    #
    # # weights: a list of numpy arrays
    # @tf.function
    # def set_weights(self, weights):
    #     tf.keras.Sequential.set_weights(self, weights)
    # @tf.function(input_signature=[])
    # def save_weights(self):
    #     tf.keras.Sequential.save_weights(self, "TODO", overwrite=True, save_format="tf")
    #
    # @tf.function
    # def load_weights(self, filepath):
    #     tf.keras.Sequential.load_weights(self, filepath)


# TODO model function to persist model with current values
# Official python guide: https://keras.io/guides/serialization_and_saving/
# TODO try that one: https://github.com/tensorflow/rust/issues/279#issuecomment-749339129

model = QLearningModel()
model.summary()

# TODO find a (fast) way to create a copy of the model to refresh the target model after a batch run
model.save('q_learning_model_1',
           save_format='tf',
           signatures={
               'predict_single': model.predict_single,
               'train_model': model.train_model,
               'write_checkpoint': model.write_checkpoint
           })
