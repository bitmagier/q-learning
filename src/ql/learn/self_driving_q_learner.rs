use std::path::Path;
use std::rc::Rc;

use rand::prelude::*;

use crate::ql::learn::replay_buffer::ReplayBuffers;
use crate::ql::model::q_learning_model1::{BATCH_SIZE, QLearningModel1};
use crate::ql::prelude::{Action, Environment};

use super::misc::Immutable;

pub struct Parameter {
    /// Discount factor for past rewards
    pub gamma: f32,
    /// Maximum epsilon greedy parameter
    pub epsilon_max: f32,
    /// Minimum epsilon greedy parameter
    pub epsilon_min: f32,
    pub max_steps_per_episode: usize,
    // Number of frames to take random action and observe output
    pub epsilon_random_frames: usize,
    // Number of frames for exploration
    pub epsilon_greedy_frames: f32,
    // Maximum replay length
    // Note from python reference code: The Deepmind paper suggests 1000000 however this causes memory issues
    pub step_history_buffer_len: usize,
    // this determines directly the number of recent goal-achieving episodes required to consider the learning task done
    pub episode_reward_history_buffer_len: usize,
    // Train the model after 4 actions
    pub update_after_actions: usize,
    // After how many frames we want to update the target network
    pub update_target_network_after_num_frames: usize,
    pub stats_after_steps: usize
}

impl Parameter {
    fn epsilon_interval(&self) -> f32 {
        self.epsilon_max - self.epsilon_min
    }
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon_max: 1.0,
            epsilon_min: 0.1,
            max_steps_per_episode: 10000,
            epsilon_random_frames: 50000,
            epsilon_greedy_frames: 1000000.0,
            step_history_buffer_len: 100000,
            episode_reward_history_buffer_len: 100,
            update_after_actions: 4,
            update_target_network_after_num_frames: 10000,
            stats_after_steps: 20
        }
    }
}

/**
    A self-driving Q learning algorithm.
    It's directly connected to a (Game-) Environment and drives the speed of the steps in that environment with it's response.

    You may find some basic fundamentals of Reinforcement learning useful. Here is a short wrapup in the sources under 'analysis/reinforcement_learning.md'.

    (Original python script)[https://keras.io/examples/rl/deep_q_network_breakout/]
```python
  while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
 ```
 */
pub struct SelfDrivingQLearner<E: Environment> {
    param: Immutable<Parameter>,
    model: QLearningModel1<E>,
    model_target: QLearningModel1<E>,
    checkpoint_file: String,
    environment: E,
}

impl<E: Environment> SelfDrivingQLearner<E> {
    pub fn from_scratch(environment: E, param: Parameter, checkpoint_file: &Path) -> Self {
        let checkpoint_file = checkpoint_file.to_str()
            .expect("file name should have a UTF-8 compatible path")
            .to_owned();
        Self {
            param: Immutable::new(param),
            model: QLearningModel1::init(),
            model_target: QLearningModel1::init(),
            checkpoint_file,
            environment,
        }
    }

    pub fn from_checkpoint(environment: E, param: Parameter, checkpoint_file: &Path) -> Self {
        let checkpoint_file = checkpoint_file.to_str()
            .expect("file name should have a UTF-8 compatible path")
            .to_owned();
        let model = QLearningModel1::init();
        model.read_checkpoint(&checkpoint_file);
        let model_target = QLearningModel1::init();
        model_target.read_checkpoint(&checkpoint_file);

        Self {
            param: Immutable::new(param),
            model,
            model_target,
            checkpoint_file,
            environment,
        }
    }

    pub fn learn_until_mastered(&mut self) {
        let mut replay_buffers = ReplayBuffers::new(self.param.step_history_buffer_len, self.param.episode_reward_history_buffer_len);

        let mut step_count: usize = 0;
        let mut episode_count: usize = 0;
        let mut running_reward: f32 = 0.0;

        ///  Epsilon greedy parameter
        let mut epsilon: f32 = self.param.epsilon_max;

        loop {
            self.environment.reset();
            let (mut state, _, _) = self.environment.step(E::no_action());

            let mut episode_reward: f32 = 0.0;

            for _ in [0..self.param.max_steps_per_episode] {
                step_count += 1;

                // Use epsilon-greedy for exploration
                let action: E::Action =
                    if step_count < self.param.epsilon_random_frames
                        || epsilon > thread_rng().gen::<f32>() {
                        // Take random action
                        let a = thread_rng().gen_range(0..<E as Environment>::Action::ACTION_SPACE);
                        Action::try_from_numeric(a).unwrap()
                    } else {
                        // Predict best action Q-values from environment state
                        self.model.predict_action(&state)
                    };

                // Decay probability of taking random action
                epsilon = f32::max(
                    epsilon - self.param.epsilon_interval() / self.param.epsilon_greedy_frames,
                    self.param.epsilon_min,
                );

                // Apply the sampled action in our environment
                let (state_next, reward, done) = self.environment.step(action);

                episode_reward += reward;

                // Save actions and states in replay buffer
                replay_buffers.add_step_items(action, Rc::clone(&state), Rc::clone(&state_next), reward, done);
                state = state_next;

                // Update every fourth frame and once batch size is over 32
                if step_count % self.param.update_after_actions == 0 && replay_buffers.done_history.len() > BATCH_SIZE {
                    // Get indices of samples for replay buffers
                    let indices: [usize; BATCH_SIZE] = {
                        let range = replay_buffers.done_history.len();
                        (0..BATCH_SIZE)
                            .map(|_| thread_rng().gen_range(0..range))
                            .collect::<Vec<usize>>().try_into().unwrap()
                    };

                    let state_samples = replay_buffers.state_history.get_many(&indices);
                    let state_next_samples = replay_buffers.state_next_history.get_many(&indices);
                    let reward_samples = replay_buffers.reward_history.get_many_as_val(&indices);
                    let action_samples = replay_buffers.action_history.get_many_as_val(&indices);
                    let done_samples = replay_buffers.done_history.get_many_as_val(&indices).map(|e| bool_to_f32(e));

                    // Build the updated Q-values for the sampled future states
                    // Use the target model for stability
                    let future_rewards = self.model_target.batch_predict_future_reward(state_next_samples);
                    // Q value = reward + discount factor * expected future reward
                    let updated_q_values = array_add(&reward_samples, &array_mul(future_rewards, self.param.gamma));

                    // If final frame set the last value to -1
                    let updated_q_values: [f32; BATCH_SIZE] = updated_q_values.iter().zip(done_samples.iter())
                        .map(|(updated_q_value, done)| updated_q_value * (1.0 - done) - done)
                        .collect::<Vec<_>>()
                        .try_into().unwrap();

                    let loss = self.model.train(state_samples, action_samples, updated_q_values);

                    if step_count % self.param.stats_after_steps == 0 {
                        log::debug!("step: {}, episode: {}, running_reward: {}, training loss: {}, ", step_count, episode_count, running_reward, loss);
                    }

                    if step_count % self.param.update_target_network_after_num_frames == 0 {
                        // update the target network with new weights
                        self.model.write_checkpoint(&self.checkpoint_file);
                        self.model_target.read_checkpoint(&self.checkpoint_file);
                        log::info!("running reward: {:.2} at episode {}, step count (frames): {}", running_reward, episode_count, step_count);
                    }

                    if done {
                        break;
                    }
                }
            }

            // Update running reward to check condition for solving
            replay_buffers.add_episode_reward(episode_reward);
            running_reward = replay_buffers.avg_episode_rewards();
            episode_count += 1;

            if running_reward >= E::total_reward_goal() {
                log::info!("Solved at episode {}!", episode_count);
                break;
            }
        }
    }
}

fn bool_to_f32(v: bool) -> f32 {
    match v {
        true => 1.0,
        false => 0.0
    }
}

fn array_add<const N: usize>(lhs: &[f32; N], rhs: &[f32; N]) -> [f32; N] {
    lhs.iter().zip(rhs.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>().try_into().unwrap()
}

fn array_mul<const N: usize>(slice: [f32; N], value: f32) -> [f32; N] {
    slice.map(|e| e * value)
}

