import tensorflow as tf
from keras import layers, optimizers

# These are the top references to follow while finding a suitable implementation:
# https://keras.io/examples/rl/deep_q_network_breakout/
# https://github.com/tensorflow/rust/tree/master/examples

NUM_ACTIONS = 4
BATCH_SIZE = 32  # Size of batch taken from replay buffer


class QLearningModel(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super(QLearningModel, self).__init__(*args, **kwargs)
        # model.add(keras.Input(shape=(4,)))
        self.add(layers.Conv2D(32, 12, strides=6, activation='relu', input_shape=(600, 800, 3), name='convolution_layer1'))
        self.add(layers.Conv2D(64, 6, strides=3, activation='relu', name='convolution_layer2'))
        self.add(layers.Conv2D(64, 4, strides=2, activation='relu', name='convolution_layer3'))
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(512, activation='relu', name='full_layer'))
        self.add(layers.Dense(NUM_ACTIONS, activation='linear', name='action_layer'))

    # Predict action from environment state
    @tf.function
    def predict_single(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0])
        return action

    @tf.function
    def train_model(self, state_samples, action_samples, updated_q_values):
        # Create a mask - so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_samples, NUM_ACTIONS)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_samples)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return {'loss': loss}

    # def rl_step_first_predict_action(self, state):
    #     self.frame_count += 1
    #
    #     # Use epsilon-greedy for exploration
    #     if self.frame_count < EPSILON_RANDOM_FRAMES or self.epsilon > np.random.rand(1)[0]:
    #         # Take random action
    #         action = np.random.choice(NUM_ACTIONS)
    #     else:
    #         => predict_single(state)
    #         # Decay probability of taking random action
    #     self.epsilon -= EPSILON_INTERVAL / EPSILON_GREEDY_FRAMES
    #     self.epsilon = max(self.epsilon, EPSILON_MIN)
    #
    #     return action

    # # Call after the predicted action was applied in our environment
    # #    e.g. after something like this: state_next, reward, done = environment.step(action)
    # @tf.function
    # def rl_step_second_learn_from_applied_action(self, state, action, state_next, reward, done):
    #     solved = False
    #
    #     state_next = np.array(state_next)
    #     self.episode_reward += reward
    #
    #     # Save actions and states in replay buffer
    #     self.action_history.append(action)
    #     self.state_history.append(state)
    #     self.state_next_history.append(state_next)
    #     self.done_history.append(done)
    #     self.rewards_history.append(reward)
    #
    #     # Update every fourth frame and once batch size is over 32
    #     if self.frame_count % UPDATE_AFTER_ACTIONS == 0 and len(self.done_history) > BATCH_SIZE:
    #         # Get indices of samples for replay buffers
    #         indices = np.random.choice(range(len(self.done_history)), size=BATCH_SIZE)
    #
    #         # Using list comprehension to sample from replay buffer
    #         state_sample = np.array([self.state_history[i] for i in indices])
    #         state_next_sample = np.array([self.state_next_history[i] for i in indices])
    #         rewards_sample = [self.rewards_history[i] for i in indices]
    #         action_sample = [self.action_history[i] for i in indices]
    #         done_sample = tf.convert_to_tensor(
    #             [float(self.done_history[i]) for i in indices]
    #         )
    #
    #         # Build the updated Q-values for the sampled future states
    #         # Use the target model for stability
    #         # => predict_batch
    #         future_rewards = self.model_target.predict(state_next_sample)
    #         # Q value = reward + discount factor * expected future reward
    #         updated_q_values = rewards_sample + GAMMA * tf.reduce_max(
    #             future_rewards, axis=1
    #         )
    #
    #         # If final frame set the last value to -1
    #         updated_q_values = updated_q_values * (1 - done_sample) - done_sample
    #
    #         # => train_model()
    #         # # Create a mask - so we only calculate loss on the updated Q-values
    #         # masks = tf.one_hot(action_sample, NUM_ACTIONS)
    #         #
    #         # with tf.GradientTape() as tape:
    #         #     # Train the model on the states and updated Q-values
    #         #     q_values = self.model(state_sample)
    #         #
    #         #     # Apply the masks to the Q-values to get the Q-value for action taken
    #         #     q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
    #         #     # Calculate loss between new Q-value and old Q-value
    #         #     loss = self.loss_function(updated_q_values, q_action)
    #         #
    #         # # Backpropagation
    #         # grads = tape.gradient(loss, self.model.trainable_variables)
    #         # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #     if self.frame_count % UPDATE_TARGET_NETWORK == 0:
    #         # update the target network with new weights
    #         self.model_target.set_weights(self.model.get_weights())
    #         # Log details
    #         template = "running reward: {:.2f} at episode {}, frame count {}"
    #         print(template.format(self.running_reward, self.episode_count, self.frame_count))
    #
    #     # Limit the state and reward history
    #     if len(self.rewards_history) > MAX_MEMORY_LENGTH:
    #         del self.rewards_history[:1]
    #         del self.state_history[:1]
    #         del self.state_next_history[:1]
    #         del self.action_history[:1]
    #         del self.done_history[:1]
    #
    #     if done:
    #         return solved
    #
    #     if self.episode_step > MAX_STEPS_PER_EPISODE:
    #         # Update running reward to check condition for solving
    #         self.episode_reward_history.append(self.episode_reward)
    #         if len(self.episode_reward_history) > EPISODE_REWARD_HISTORY_MAX_LENGTH:
    #             del self.episode_reward_history[:1]
    #         self.running_reward = np.mean(self.episode_reward_history)
    #
    #         self.episode_count += 1
    #
    #         if self.running_reward > CUMULATIVE_REWARD_GOAL:
    #             print("Solved at episode {}!".format(self.episode_count))
    #             solved = True
    #     return solved


model = QLearningModel()
model.compile(optimizer=optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
              # Using huber loss for stability
              loss=tf.keras.losses.Huber(),
              metrics=tf.keras.metrics.Accuracy())

model.summary()

t_state = tf.TensorSpec(shape=[600, 800, 3], dtype=tf.float32, name='state')

# TODO export train_model

model.save('q_learning_keras_model',
           save_format='tf',
           signatures={
               'predict_single': model.predict_single.get_concrete_function(t_state),
               # 'predict': model.predict.get_concrete_function(),
               # 'get_weights': model.get_weights,
               # 'set_weights': model.set_weights
           })

# TODO save target model

# one way to get output names via saved_model_cli:
# saved_model_cli show --dir /path/to/saved-model/ --all
