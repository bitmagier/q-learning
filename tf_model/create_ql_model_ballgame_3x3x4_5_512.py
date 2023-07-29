import keras
import tensorflow as tf
from keras import layers, optimizers, losses

BATCH_SIZE = 512
INPUT_SIZE_X = 3
INPUT_SIZE_Y = 3
INPUT_CHANNELS = 4
ACTION_SPACE = 5


class QLearningModel_BallGame_3x3x4_5_512(tf.keras.Sequential):

    def __init__(self, *args, **kwargs):
        super(QLearningModel_BallGame_3x3x4_5_512, self).__init__(*args, **kwargs)
        self.add(tf.keras.Input(shape=(INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS,)))

        # self.add(layers.Dense(128, activation='sigmoid'))
        # self.add(layers.Dense(128, activation='sigmoid'))
        # self.add(layers.Dense(128, activation='sigmoid'))

        self.add(layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation="relu"))
        self.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation="relu"))
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(units=256, activation="relu"))

        # activation function should be linear, to provide a value-range matching the Q-value-range
        self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))

        # learning_rate = keras.optimizers.learning_rate_schedule.PolynomialDecay(
        #       initial_learning_rate=0.001,
        #       decay_steps=1_000_000,
        #       end_learning_rate=0.0002,
        #       power=2.0)

        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00015, clipvalue=1.0),
            loss=keras.losses.MeanSquaredError(),
            metrics=['accuracy'],
            jit_compile=True
        )

    # Predict action from environment state
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS], dtype=tf.float32, name='state')
    ])
    def predict_action(self, state):
        state_tensor = tf.expand_dims(state, 0)
        action_probs = self.call(state_tensor)
        # Take best action
        action = tf.argmax(action_probs[0])
        return {'action': action}

    # WTF: the batch-optimized predict() function can not be exported or wrapped in a user function.
    # So we have to stick with call()
    # => Maybe we try a model based on tf.Module instead of tf.keras.Model
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS], dtype=tf.float32,
                      name='state_batch')
    ])
    def batch_predict_max_future_reward(self, state_batch):
        reward_batch = tf.reduce_max(self.call(state_batch), axis=1)
        return {'reward_batch': reward_batch}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS], dtype=tf.float32,
                      name='state_batch'),
        tf.TensorSpec(shape=[BATCH_SIZE, ACTION_SPACE], dtype=tf.float32, name='action_batch_one_hot'),
        tf.TensorSpec(shape=[BATCH_SIZE], dtype=tf.float32, name='updated_q_values')
    ])
    def train_model(self, state_batch, action_batch_one_hot, updated_q_values):
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.call(state_batch)
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, action_batch_one_hot), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.trainable_variables)
        _ = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='file')])
    def write_checkpoint(self, file):
        checkpoint = tf.train.Checkpoint(self)
        out = checkpoint.write(file)
        return {'file': tf.convert_to_tensor(out)}

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='file')])
    def read_checkpoint(self, file):
        checkpoint = tf.train.Checkpoint(self)
        file = tf.get_static_value(file)
        status = checkpoint.read(file)
        if file is not None:
            status.assert_consumed()
        return {'dummy': tf.constant("")}


model = QLearningModel_BallGame_3x3x4_5_512()
model.summary()

model.save('saved/ql_model_ballgame_3x3x4_5_512',
           save_format='tf',
           signatures={
               'predict_action': model.predict_action,
               'batch_predict_max_future_reward': model.batch_predict_max_future_reward,
               'train_model': model.train_model,
               'write_checkpoint': model.write_checkpoint,
               'read_checkpoint': model.read_checkpoint,
           })
