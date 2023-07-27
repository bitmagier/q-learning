use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use itertools::Itertools;
use num_format::{CustomFormat, Grouping, ToFormattedString};
use rand::distributions::Uniform;
use rand::prelude::*;
use rustc_hash::FxHashMap;

use crate::ql::learn::replay_buffer::ReplayBuffer;
use crate::ql::prelude::{Action, DebugVisualizer, Environment, QLearningModel};
use crate::util::dbscan;
use crate::util::immutable::Immutable;

pub struct Parameter {
    /// Discount factor for past rewards
    pub gamma: f32,
    /// Maximum epsilon greedy parameter
    pub epsilon_max: f64,
    /// Minimum epsilon greedy parameter
    pub epsilon_min: f64,
    pub max_steps_per_episode: usize,
    // Number of frames to take only random action and observe output
    pub epsilon_pure_random_steps: usize,
    // Number of frames for exploration
    pub epsilon_greedy_steps: f64,
    // Maximum replay length
    // Note from python reference code: The Deepmind paper suggests 1000000 however this causes memory issues
    pub history_buffer_len: usize,
    // Train the model after n actions
    pub update_after_actions: usize,
    // After how many frames we want to update the target network
    pub update_target_network_after_num_steps: usize,
    // this determines directly the number of recent goal-achieving episodes required to consider the learning task done
    pub episode_reward_history_buffer_len: usize,
    pub stats_after_steps: usize,
    // Percentage of total reward goal, which any single episode needs to reach (regardless of the average reward) 
    pub lowest_episode_reward_goal_threshold_pct: f32
}

impl Parameter {
    fn epsilon_interval(&self) -> f64 {
        self.epsilon_max - self.epsilon_min
    }
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon_max: 1.0,
            epsilon_min: 0.1,
            max_steps_per_episode: 10_000,
            epsilon_pure_random_steps: 50_000,
            epsilon_greedy_steps: 1_000_000.0,
            history_buffer_len: 1_000_000,
            update_after_actions: 4,
            update_target_network_after_num_steps: 10_000,
            episode_reward_history_buffer_len: 100,
            stats_after_steps: 10_000,
            lowest_episode_reward_goal_threshold_pct: 0.9
        }
    }
}

// TODO Add TEST without an AI model, but with a Q-tabular  

pub struct SelfDrivingQLearner<E, M, const BATCH_SIZE: usize>
where
    E: Environment,
    M: QLearningModel<BATCH_SIZE, E=E>,
{
    environment: Arc<RwLock<E>>,
    param: Immutable<Parameter>,
    model: M,
    // "target_model"
    stabilized_model: M,
    checkpoint_file: String,
    replay_buffer: ReplayBuffer<Rc<E::S>, E::A>,
    step_count: usize,
    episode_count: usize,
    running_reward: f32,
    ///  Epsilon greedy parameter
    epsilon: f64,
}

impl<E, M, const BATCH_SIZE: usize> SelfDrivingQLearner<E, M, BATCH_SIZE>
where
    E: Environment,
    M: QLearningModel<BATCH_SIZE, E=E>,
{
    pub fn new(environment: Arc<RwLock<E>>, param: Parameter, model_instance1: M, model_instance2: M, checkpoint_file: &Path) -> Self {
        let checkpoint_file = checkpoint_file
            .to_str()
            .expect("file name should have a UTF-8 compatible path")
            .to_owned();
        let replay_buffer = ReplayBuffer::new(param.history_buffer_len, param.episode_reward_history_buffer_len);
        let epsilon = param.epsilon_max;

        Self {
            environment,
            param: Immutable::new(param),
            model: model_instance1,
            stabilized_model: model_instance2,
            checkpoint_file,
            replay_buffer,
            step_count: 0,
            episode_count: 0,
            running_reward: 0.0,
            epsilon,
        }
    }

    pub fn load_model_checkpoint(&mut self, checkpoint_file: &Path) {
        let checkpoint_file = checkpoint_file.to_str().expect("file name should have a UTF-8 compatible path");
        self.model.read_checkpoint(checkpoint_file);
        self.stabilized_model.read_checkpoint(checkpoint_file);
    }

    pub fn learn_till_mastered(&mut self) -> Result<()> {
        while !self.solved() {
            self.learn_episode()?
        }
        Ok(())
    }

    pub fn solved(&self) -> bool {
        let env = self.environment.read().unwrap();
        if self.running_reward >= env.episode_reward_goal_mean()
            && self.replay_buffer.min_episode_reward() >= env.episode_reward_goal_mean() * self.param.lowest_episode_reward_goal_threshold_pct
        {
            true
        } else {
            false
        }
    }

    pub fn learn_episode(&mut self) -> Result<()> {
        self.environment.write().unwrap().reset();

        let mut state = self.environment.read().unwrap().state_as_rc();
        log::trace!("started learning episode {}", self.episode_count);

        let mut episode_reward: f32 = 0.0;

        for _ in 0..self.param.max_steps_per_episode {
            self.step_count += 1;

            // Use epsilon-greedy for exploration
            let action: E::A =
                if self.step_count < self.param.epsilon_pure_random_steps || self.epsilon > thread_rng().gen_range(0_f64..1_f64) {
                    // Take random action
                    let a = thread_rng().gen_range(0..E::A::ACTION_SPACE);
                    Action::try_from_numeric(a)?
                } else {
                    // Predict best action Q-values from environment state
                    self.model.predict_action(&state)
                };

            // Decay probability of taking random action
            self.epsilon = f64::max(
                self.epsilon - self.param.epsilon_interval() / self.param.epsilon_greedy_steps,
                self.param.epsilon_min,
            );

            log::trace!("{}", state.one_line_info());
            // Apply the sampled action in our environment
            let (state_next, reward, done) = self.environment.write().unwrap().step_as_rc(action);
            log::trace!("step with action {} resulted in reward: {:.2}, done: {}", action, reward, done);

            episode_reward += reward;

            // Save actions and states in replay buffer
            self.replay_buffer.add(action, state, Rc::clone(&state_next), reward, done);
            state = state_next;

            // Update every n-th step (e.g. fourth frame), once the replay buffer is beyond BATCH_SIZE
            if self.step_count % self.param.update_after_actions == 0 && self.replay_buffer.len() > BATCH_SIZE {
                // Get indices of samples for replay buffers
                let indices: [usize; BATCH_SIZE] = generate_distinct_random_ids(0..self.replay_buffer.len());

                let replay_samples = self.replay_buffer.get_many(&indices);

                // Build the updated Q-values for the sampled future states
                // Use the target model for stability
                let max_future_rewards = self.stabilized_model.batch_predict_max_future_reward(replay_samples.state_next);

                // Q value = reward + discount factor * expected future reward
                let mut updated_q_values = add_arrays(&replay_samples.reward, &array_mul(max_future_rewards, self.param.gamma));

                // for terminal steps, the updated q-value shall be exactly the reward (see deepmind paper)
                for (i, _) in replay_samples.state.iter().enumerate() {
                    if replay_samples.done[i] {
                        updated_q_values[i] = replay_samples.reward[i]
                    }
                }

                let loss = self.model.train(replay_samples.state, replay_samples.action, updated_q_values)?;
                if log::log_enabled!(log::Level::Trace) || self.step_count % self.param.stats_after_steps == 0 {
                    log::debug!("training step: loss: {}", loss)
                }
            }

            if log::log_enabled!(log::Level::Trace) || self.step_count % self.param.stats_after_steps == 0 {
                log::debug!(
                    "step: {}, episode: {}, running_reward: {}",
                    self.step_count,
                    self.episode_count,
                    self.running_reward
                );
            }

            if self.step_count % self.param.update_target_network_after_num_steps == 0 {
                // update the target network with new weights
                self.model.write_checkpoint(&self.checkpoint_file);
                self.stabilized_model.read_checkpoint(&self.checkpoint_file);
                self.learning_update_log();
            }

            if done {
                break;
            }
        }

        // Update running reward to check condition for solving
        self.replay_buffer.add_episode_reward(episode_reward);
        if self.episode_count >= self.param.episode_reward_history_buffer_len {
            self.running_reward = self.replay_buffer.avg_episode_reward();
        }
        self.episode_count += 1;

        if self.solved() {
            self.learning_update_log()
        }
        
        Ok(())
    }

    fn learning_update_log(&self) {
        let number_format = number_format();
        
        let num_rewards = self.replay_buffer.episode_rewards().len();
        let episode_rewards = self.replay_buffer.episode_rewards();
        let reward_distribution = dbscan::cluster_analysis(&episode_rewards, 0.35, num_rewards / 50);

        let mut action_counts = FxHashMap::<E::A, usize>::default();
        for &a in &self.replay_buffer.actions().buffer {
            action_counts.entry(a).and_modify(|e| *e += 1).or_insert(1);
        }

        let total_actions = self.replay_buffer.actions().buffer.len();
        let action_distribution_line = action_counts.iter()
            .map(|(&action, &count)| {
                let ratio = 100.0 * count as f32 / total_actions as f32;
                format!("{} {:.1}%", action, ratio)
            }).join(", ");
        
        log::info!("\n\
    episode {}, step count: {}, epsilon: {:.2}, current rewards: (mean: {:.1}, low: {:.1}), reward goal: (mean: {:.1}, low: {:.1})\n\
    reward distribution: {}\n\
    action distribution (last {}): {}",
            self.episode_count.to_formatted_string(&number_format),
            self.step_count.to_formatted_string(&number_format),
            self.epsilon,
            self.replay_buffer.avg_episode_reward(),
            self.replay_buffer.min_episode_reward(),
            self.environment.read().unwrap().episode_reward_goal_mean(),
            self.environment.read().unwrap().episode_reward_goal_mean() * self.param.lowest_episode_reward_goal_threshold_pct,
            reward_distribution,
            total_actions.to_formatted_string(&number_format), 
            action_distribution_line
        );
    }
}

fn generate_distinct_random_ids<const BATCH_SIZE: usize>(range: Range<usize>) -> [usize; BATCH_SIZE] {
    assert!(range.end - range.start >= BATCH_SIZE);
    let mut result = [0_usize; BATCH_SIZE];

    let distribution = Uniform::from(range);
    let mut rng = thread_rng();

    for i in 0..BATCH_SIZE {
        result[i] = loop {
            let x = distribution.sample(&mut rng);
            if !result[0..i].contains(&x) {
                break x;
            }
        }
    }
    result
}

fn add_arrays<const N: usize>(lhs: &[f32; N], rhs: &[f32; N]) -> [f32; N] {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn array_mul<const N: usize>(slice: [f32; N], value: f32) -> [f32; N] {
    slice.map(|e| e * value)
}

fn number_format() -> CustomFormat {
    CustomFormat::builder()
        .grouping(Grouping::Standard)
        .minus_sign("-")
        .separator("_")
        .build()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use crate::ql::ballgame_test_environment::BallGameTestEnvironment;
    use crate::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};

    use super::*;

    #[test]
    fn test_learner_single_episode() -> Result<()> {
        let param = Parameter::default();
        let model_init = || QLearningTensorflowModel::<BallGameTestEnvironment>::load(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);
        let model_instance1 = model_init();
        let model_instance2 = model_init();
        let checkpoint_file = tempfile::tempdir().unwrap().into_path().join("test_learner_ckpt");
        let environment = Arc::new(RwLock::new(BallGameTestEnvironment::default()));
        let mut learner = SelfDrivingQLearner::new(environment, param, model_instance1, model_instance2, &checkpoint_file);
        assert!(!learner.solved());

        learner.learn_episode()?;

        assert!(!learner.solved());
        assert!(learner.step_count > 1);
        assert_eq!(learner.episode_count, 1);

        Ok(())
    }

    #[test]
    fn test_100x_generate_distinct_random_ids() {
        for _ in 0..100 {
            test_generate_distinct_random_ids();
        }
    }

    #[test]
    fn test_generate_distinct_random_ids() {
        let result: [usize; 50] = generate_distinct_random_ids(0..100);
        let mut r = Vec::from(result);
        r.sort();
        r.dedup();
        assert_eq!(r.len(), 50);
        assert!(r.iter().all(|e| (0..100).contains(e)));
    }
}
