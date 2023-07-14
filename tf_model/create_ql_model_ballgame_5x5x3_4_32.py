import keras
import tensorflow as tf
from keras import layers, optimizers, losses

INPUT_SIZE_X = 5
INPUT_SIZE_Y = 5
INPUT_LAYERS = 4
ACTION_SPACE = 4
BATCH_SIZE = 32


class QLearningModel_BallGame_5x5x4_4_32(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super(QLearningModel_BallGame_5x5x4_4_32, self).__init__(*args, **kwargs)
        self.add(tf.keras.Input(shape=(INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS,)))
        self.add(layers.Conv2D(16, 3, strides=1, data_format='channels_last', activation='relu', name='convolution_layer1'))
        self.add(layers.Conv2D(32, 2, strides=1, activation='relu', name='convolution_layer2'))
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(256, activation='relu', name='full_layer1'))
        self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))

        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                     # Using huber loss for stability
                     loss=keras.losses.Huber(),
                     metrics=['accuracy'],
                     jit_compile=True
                     )

    # Predict action from environment state
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32, name='state')])
    def predict_action(self, state):
        state_tensor = tf.expand_dims(state, 0)
        action_probs = self.call(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0])
        return {'action': action}

    # WTF: the batch-optimized predict() function can not be exported or wrapped in a user function.
    # So we have to stick with call()
    # => Maybe we try a model based on tf.Module instead of tf.keras.Model
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32,
                      name='state_batch')])
    def batch_predict_max_future_reward(self, state_batch):
        reward_batch = tf.reduce_max(self.call(state_batch), axis=1)
        return {'reward_batch': reward_batch}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32,
                      name='state_batch'),
        tf.TensorSpec(shape=[BATCH_SIZE, 1], dtype=tf.uint8, name='action_batch'),
        tf.TensorSpec(shape=[BATCH_SIZE, 1], dtype=tf.float32, name='updated_q_values')
    ])
    def train_model(self, state_batch, action_batch, updated_q_values):
        # Create a mask - so we only calculate loss on the updated Q-values
        masks = tf.one_hot(indices=action_batch, depth=ACTION_SPACE)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.call(state_batch)
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='file')])
    def write_checkpoint(self, file):
        checkpoint = tf.train.Checkpoint(self)
        file = checkpoint.write(file)
        return {'file': tf.convert_to_tensor(file)}

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='file')])
    def read_checkpoint(self, file):
        checkpoint = tf.train.Checkpoint(self)
        file = tf.get_static_value(file)
        checkpoint.read(file)
        return {'dummy': tf.constant("")}


model = QLearningModel_BallGame_5x5x4_4_32()
model.summary()

model.save('saved/ql_model_ballgame_5x5x4_4_32',
           save_format='tf',
           signatures={
               'predict_action': model.predict_action,
               'batch_predict_max_future_reward': model.batch_predict_max_future_reward,
               'train_model': model.train_model,
               'write_checkpoint': model.write_checkpoint,
               'read_checkpoint': model.read_checkpoint
           })