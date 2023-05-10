import keras
import numpy as np
import tensorflow as tf
from keras import layers, optimizers

# These are the top references to follow while finding a suitable implementation:
# https://keras.io/examples/rl/deep_q_network_breakout/


NUM_ACTIONS = 4

# Configuration parameters for the whole setup
SEED = 42
GAMMA = 0.99  # Discount factor for past rewards
EPSILON_MIN = 0.1  # Minimum epsilon greedy parameter
EPSILON_MAX = 1.0  # Maximum epsilon greedy parameter
EPSILON_INTERVAL = (
        EPSILON_MAX - EPSILON_MIN
)  # Rate at which to reduce chance of random action being taken
BATCH_SIZE = 32  # Size of batch taken from replay buffer
MAX_STEPS_PER_EPISODE = 10000

# Number of frames to take random action and observe output
EPSILON_RANDOM_FRAMES = 50000
# Number of frames for exploration
EPSILON_GREEDY_FRAMES = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
MAX_MEMORY_LENGTH = 100000
# Train the model after 4 actions
UPDATE_AFTER_ACTIONS = 4
# How often to update the target network
UPDATE_TARGET_NETWORK = 10000  # TODO precise name

EPISODE_REWARD_HISTORY_MAX_LENGTH = 100

# Target to consider the task solved
CUMULATIVE_REWARD_GOAL = 40


def create_q_model():
    # input: batches of normalized 2D Game screenshots (`tf.random.normal`)
    # effective input_shape: (batch-size, x, y, color-channels)

    # filters: integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    #          initial filters in a network is responsible for detecting edges and blobs; 16 or 32 is recommended
    # kernel_size

    model = keras.Sequential([
        layers.Conv2D(32, 12, strides=6, activation='relu', input_shape=(600, 800, 3), name='convolution_layer1'),
        layers.Conv2D(64, 6, strides=3, activation='relu', name='convolution_layer2'),
        layers.Conv2D(64, 4, strides=2, activation='relu', name='convolution_layer3'),
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='full_layer'),
        layers.Dense(NUM_ACTIONS, activation='linear', name='action_layer')
    ])
    model.summary()
    return model


class BreakoutQLearningCore:
    def __init__(self, *args, **kwargs):
        # The first model makes the predictions for Q-values which are used to make a action.
        self.model = create_q_model()
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.model_target = create_q_model()

        self.optimizer = optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        # Using huber loss for stability
        self.loss_function = tf.keras.losses.Huber()

        self.episode_count = 0
        self.episode_reward = 0
        self.episode_step = 0

        self.epsilon = 1.0  # Epsilon greedy parameter

        self.running_reward = 0
        self.frame_count = 0

        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []

    @tf.function
    def rl_step_first_predict_action(self, state):
        self.frame_count += 1

        # Use epsilon-greedy for exploration
        if self.frame_count < EPSILON_RANDOM_FRAMES or self.epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(NUM_ACTIONS)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
        self.epsilon -= EPSILON_INTERVAL / EPSILON_GREEDY_FRAMES
        self.epsilon = max(self.epsilon, EPSILON_MIN)

        return action

    # Call after the predicted action was applied in our environment
    #    e.g. after something like this: state_next, reward, done = environment.step(action)
    @tf.function
    def rl_step_second_learn_from_applied_action(self, state, action, state_next, reward, done):
        solved = False

        state_next = np.array(state_next)
        self.episode_reward += reward

        # Save actions and states in replay buffer
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)

        # Update every fourth frame and once batch size is over 32
        if self.frame_count % UPDATE_AFTER_ACTIONS == 0 and len(self.done_history) > BATCH_SIZE:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self.done_history)), size=BATCH_SIZE)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self.model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + GAMMA * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask - so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, NUM_ACTIONS)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.frame_count % UPDATE_TARGET_NETWORK == 0:
            # update the target network with new weights
            self.model_target.set_weights(self.model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(self.running_reward, self.episode_count, self.frame_count))

        # Limit the state and reward history
        if len(self.rewards_history) > MAX_MEMORY_LENGTH:
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

        if done:
            return solved

        if self.episode_step > MAX_STEPS_PER_EPISODE:
            # Update running reward to check condition for solving
            self.episode_reward_history.append(self.episode_reward)
            if len(self.episode_reward_history) > EPISODE_REWARD_HISTORY_MAX_LENGTH:
                del self.episode_reward_history[:1]
            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1

            if self.running_reward > CUMULATIVE_REWARD_GOAL:
                print("Solved at episode {}!".format(self.episode_count))
                solved = True
        return solved


# example: https://www.tensorflow.org/tutorials/images/cnn
# another example: https://towardsdatascience.com/convolutional-neural-networks-with-tensorflow-2d0d41382d32


# (1)[https://www.tensorflow.org/guide/keras/sequential_model]
# (2)[https://keras.io/guides/sequential_model/#:~:text=A%20Sequential%20model%20is%20appropriate,tensor%20and%20one%20output%20tensor.&text=A%20Sequential%20model%20is%20not,multiple%20inputs%20or%20multiple%20outputs]
# (3)[https://pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/]


# TODO optimization algorithms Stochastic Gradient Descent (SGD), RMSprop etc


# TODO define prediction function (single input/output)
# TODO define training function (batch)
#
# Standard model functions:
# fit() - Trains the model for a fixed number of epochs. This is for a predefined set of input (sliced by batch_size) and predefined number of training epochs.
#           This is designed for supervised learning. It's probably not directly suitable for us here.
#           Maybe this guide helps: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
# evaluate() - calculates the loss + metrics
# predict() - output calculation - optimized for batches = large amount of data at once
# __call__() - output calculation for a single input
#
#
# see https://www.tensorflow.org/guide/keras/train_and_evaluate/


# TODO How to save the custom class containing two models, so that we can call it from Rust?
# =>
# - Implement a pure Tensorflow Model (with 2 instances) in Python
# - Implement Q-Learning batch/sample/etc update logic completely in Rust - calling the basic model functions
# https://keras.io/examples/rl/deep_q_network_breakout/
learning_core = BreakoutQLearningCore()

# Get concrete function for the call and training method
first_predict_action = learning_core.rl_step_first_predict_action.get_concrete_function(tf.TensorSpec(shape=[600, 800, 3],
                                                                                                      dtype=tf.float32,
                                                                                                      name='inputs'))

second_learn = learning_core.rl_step_second_learn_from_applied_action.get_concrete_function(tf.TensorSpec(shape=[600, 800, 3],
                                                                                                          dtype=tf.float32,
                                                                                                          name="state"),
                                                                                            tf.TensorSpec(shape=[NUM_ACTIONS]),
                                                                                            tf.TensorSpec(shape=[600, 800, 3],
                                                                                                          dtype=tf.float32,
                                                                                                          name="state_next"),
                                                                                            tf.int32,
                                                                                            tf.bool)

learning_model.save('breakout_q_learning_core',
                    save_format='tf',
                    signatures={'first_predict_action': first_predict_action, 'second_learn': second_learn})

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
