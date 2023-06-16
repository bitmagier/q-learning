// TODO needs an independent presentation layer (decoupled from the egui event loop) for drawing the game state

use std::rc::Rc;

use rand::prelude::*;

use crate::pure_game_drawer::PureGameDrawer;
use crate::ql::breakout_environment::{Action, BreakoutEnvironment, Reward, State};
use crate::ql::environment::Environment;
use crate::ql::model::q_learning_tf_model1::{ACTION_SPACE, BATCH_SIZE, QLearningTfModel1};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

struct Parameter {
    seed: usize,
    /// Discount factor for past rewards
    gamma: f32,
    ///  Epsilon greedy parameter
    epsilon: f32,
    /// Minimum epsilon greedy parameter
    epsilon_min: f32,
    /// Maximum epsilon greedy parameter
    epsilon_max: f32,
    /// Size of batch taken from replay buffer
    batch_size: usize,
    max_steps_per_episode: usize,
    running_reward: f32,
    episode_count: usize,
    // Number of frames to take random action and observe output
    epsilon_random_frames: usize,
    // Number of frames for exploration
    epsilon_greedy_frames: f32,
    // Maximum replay length
    // Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length: usize,
    // Train the model after 4 actions
    update_after_actions: usize,
    // How often to update the target network => TODO refine description
    update_target_network: usize,
}

impl Parameter {
    fn epsilon_interval(&self) -> f32 {
        self.epsilon_max - self.epsilon_min
    }
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            seed: 42,
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.1,
            epsilon_max: 1.0,
            batch_size: 32,
            max_steps_per_episode: 10000,
            running_reward: 0.0,
            episode_count: 0,
            epsilon_random_frames: 50000,
            epsilon_greedy_frames: 1000000.0,
            max_memory_length: 100000,
            update_after_actions: 4,
            update_target_network: 10000,
        }
    }
}

/// Experience replay buffers
struct ReplayBuffer {
    action_history: Vec<Action>,
    state_history: Vec<Rc<State>>,
    state_next_history: Vec<Rc<State>>,
    reward_history: Vec<Reward>,
    done_history: Vec<bool>,
    episode_reward_history: Vec<Reward>,
}

impl ReplayBuffer {
    pub fn push(&mut self, action: Action, state: Rc<State>, state_next: Rc<State>, reward: Reward, done: bool, episode_reward: Reward) {
        self.action_history.push(action);
        self.state_history.push(state);
        self.state_next_history.push(state_next);
        self.reward_history.push(reward);
        self.done_history.push(done);
        self.episode_reward_history.push(episode_reward);
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self {
            action_history: vec![],
            state_history: vec![],
            state_next_history: vec![],
            reward_history: vec![],
            done_history: vec![],
            episode_reward_history: vec![],
        }
    }
}

/// Directly connected to GameMechanics and drives the speed of the game with it's response
pub struct SelfDrivingQLearner {
    p: Parameter,
    model: QLearningTfModel1,
    model_target: QLearningTfModel1,
}

impl SelfDrivingQLearner {
    pub fn from_scratch() -> Self {
        Self {
            p: Default::default(),
            model: QLearningTfModel1::init(),
            model_target: QLearningTfModel1::init(),
        }
    }

    pub fn from_checkpoint(checkpoint_file: &str) -> Self {
        let model = QLearningTfModel1::init();
        model.read_checkpoint(checkpoint_file);
        let target_model = QLearningTfModel1::init();
        target_model.read_checkpoint(checkpoint_file);

        Self {
            p: Parameter::default(),
            model,
            model_target: target_model,
        }
    }

    /**
      Original python (script)[https://keras.io/examples/rl/deep_q_network_breakout/]
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
    pub fn run(&mut self) {
        let drawer = Box::new(PureGameDrawer {});
        let mut environment = BreakoutEnvironment::new(drawer);
        let mut replay_buffer = ReplayBuffer::default();

        let mut frame_count: usize = 0;

        loop {
            environment.reset();
            let (mut state, _, _) = environment.step(environment.no_action());

            let mut episode_reward: Reward = 0;

            for timestamp in [1..self.p.max_steps_per_episode] {
                // TODO render attempts here if wish expressed (keypress 'r' maybe)
                frame_count += 1;

                // Use epsilon-greedy for exploration
                let action: Action =
                    if frame_count < self.p.epsilon_random_frames
                        || self.p.epsilon > thread_rng().gen::<f32>() {
                        // Take random action
                        thread_rng().gen_range(0..ACTION_SPACE)
                    } else {
                        // Predict best action Q-values from environment state
                        self.model.predict_single(&state)
                    };

                // Decay probability of taking random action
                self.p.epsilon = f32::max(
                    self.p.epsilon - self.p.epsilon_interval() / self.p.epsilon_greedy_frames,
                    self.p.epsilon_min,
                );

                // Apply the sampled action in our environment
                let (state_next, reward, done) = environment.step(action);

                episode_reward += reward;

                // Save actions and states in replay buffer
                replay_buffer.push(action, Rc::clone(&state), Rc::clone(&state_next), reward, done, episode_reward);
                state = state_next;

                // Update every fourth frame and once batch size is over 32
                if frame_count % self.p.update_after_actions == 0 && replay_buffer.done_history.len() > BATCH_SIZE {
                    // Get indices of samples for replay buffers
                    let indices: [usize; BATCH_SIZE] = {
                        let range = replay_buffer.done_history.len();
                        (0..BATCH_SIZE)
                            .map(|n| thread_rng().gen_range(0..range))
                            .collect::<Vec<usize>>().try_into().unwrap()
                    };

                    let state_samples= get_many(&replay_buffer.state_history, &indices);
                    let state_next_samples = get_many(&replay_buffer.state_next_history, &indices);
                    let reward_samples = get_many(&replay_buffer.reward_history, &indices);
                    let action_samples = get_many(&replay_buffer.action_history, &indices);
                    let done_samples = get_many(&replay_buffer.done_history, &indices);

                    // Build the updated Q-values for the sampled future states
                    // Use the target model for stability
                    let future_rewards = self.model_target.predict(state_next_samples);


                    todo!()
                }
            }
        }
    }
}

/// returns a slice of references to the values in ` vector` at the specified indices
fn get_many<'a, const N: usize, T>(slice: &'a [T], indices: &[usize; N]) -> [&'a T; N] {
    todo!()
}
