import keras
import tensorflow as tf
from keras import layers, optimizers, losses

INPUT_SIZE_X = 3
INPUT_SIZE_Y = 3
INPUT_LAYERS = 4
ACTION_SPACE = 5
BATCH_SIZE = 32

# TODO introduce a frame history dimension

class QLearningModel_BallGame_3x3x4_5_32(tf.keras.Sequential):
    loss = keras.losses.Huber()

    def __init__(self, *args, **kwargs):
        super(QLearningModel_BallGame_3x3x4_5_32, self).__init__(*args, **kwargs)
        self.add(tf.keras.Input(shape=(INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS,)))
        self.add(layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
        # self.add(layers.MaxPooling2D())
        self.add(layers.Conv2D(filters=32, kernel_size=1, activation="relu"))
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(256, activation='relu', name='full_layer1'))
        self.add(layers.Dense(256, activation='relu', name='full_layer2'))
        self.add(layers.Dense(ACTION_SPACE, activation='linear', name='action_layer'))

        learning_rate = keras.optimizers.learning_rate_schedule.PolynomialDecay(initial_learning_rate=0.01,
                                                                                decay_steps=500_000,
                                                                                end_learning_rate=0.00025,
                                                                                power=1.5)
        self.compile(
            # optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
            optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
            loss=self.loss,
            metrics=['accuracy'],
        )

    # Predict action from environment state
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32, name='state')
    ])
    def predict_action(self, state):
        state_tensor = tf.expand_dims(state, 0)
        action_probs = self.__call__(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0])
        return {'action': action}

    # WTF: the batch-optimized predict() function can not be exported or wrapped in a user function.
    # So we have to stick with call()
    # => Maybe we try a model based on tf.Module instead of tf.keras.Model
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32, name='state_batch')
    ])
    def batch_predict_max_future_reward(self, state_batch):
        reward_batch = tf.reduce_max(self.__call__(state_batch, training=False), axis=1)
        return {'reward_batch': reward_batch}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_LAYERS], dtype=tf.float32, name='state_batch'),
        tf.TensorSpec(shape=[BATCH_SIZE, ACTION_SPACE], dtype=tf.float32, name='action_batch_one_hot'),
        tf.TensorSpec(shape=[BATCH_SIZE], dtype=tf.float32, name='updated_q_values')
    ])
    def train_model(self, state_batch, action_batch_one_hot, updated_q_values):
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.__call__(state_batch, training=True)
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, action_batch_one_hot), axis=1)
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


model = QLearningModel_BallGame_3x3x4_5_32()
model.summary()

model.save('saved/ql_model_ballgame_3x3x4_5_32',
           save_format='tf',
           signatures={
               'predict_action': model.predict_action,
               'batch_predict_max_future_reward': model.batch_predict_max_future_reward,
               'train_model': model.train_model,
               'write_checkpoint': model.write_checkpoint,
               'read_checkpoint': model.read_checkpoint
           })
