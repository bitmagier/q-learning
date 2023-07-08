use std::collections::VecDeque;

pub struct ReplayBuffer<T> {
    max_buffer_len: usize,
    buffer: VecDeque<T>,
}

impl<T> ReplayBuffer<T> {
    pub fn new(max_buffer_len: usize) -> Self {
        assert!(max_buffer_len > 0);
        Self {
            max_buffer_len,
            buffer: VecDeque::with_capacity(max_buffer_len),
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn add(&mut self, element: T) {
        if (self.buffer.len() + 1) > self.max_buffer_len {
            self.buffer.pop_front().unwrap();
        }
        self.buffer.push_back(element);
    }

    /// returns a slice of references to values of `slice` at the specified `indices`
    pub fn get_many<const N: usize>(&self, indices: &[usize; N]) -> [&T; N] {
        debug_assert!(!indices.iter().any(|&e| e >= self.buffer.len()));
        indices.map(|i| &self.buffer[i])
    }
}

impl<T: Copy> ReplayBuffer<T> {
    /// returns a slice of values of `slice` at the specified `indices`
    pub fn get_many_as_val<const N: usize>(&self, indices: &[usize; N]) -> [T; N] {
        debug_assert!(!indices.iter().any(|&e| e >= self.buffer.len()));
        indices.map(|i| self.buffer[i])
    }
}

/// Experience replay buffers
pub struct ReplayBuffers<S, A>
where S: Clone,
      A: Copy
{
    action_history: ReplayBuffer<A>,
    state_history: ReplayBuffer<S>,
    state_next_history: ReplayBuffer<S>,
    reward_history: ReplayBuffer<f32>,
    done_history: ReplayBuffer<bool>,
    pub episode_reward_history: ReplayBuffer<f32>,
}

impl<S, A> ReplayBuffers<S, A>
where S: Clone,
      A: Copy
{
    pub fn new(step_buffer_len: usize, episode_reward_buffer_len: usize) -> Self {
        Self {
            action_history: ReplayBuffer::new(step_buffer_len),
            state_history: ReplayBuffer::new(step_buffer_len),
            state_next_history: ReplayBuffer::new(step_buffer_len),
            reward_history: ReplayBuffer::new(step_buffer_len),
            done_history: ReplayBuffer::new(step_buffer_len),
            episode_reward_history: ReplayBuffer::new(episode_reward_buffer_len),
        }
    }
    
    pub fn len(&self) -> usize {
        self.done_history.len()
    }

    pub fn add_step_items(&mut self, action: A, state: &S, state_next: &S, reward: f32, done: bool) {
        self.action_history.add(action);
        self.state_history.add(state.clone());
        self.state_next_history.add(state_next.clone());
        self.reward_history.add(reward);
        self.done_history.add(done);
    }

    pub fn add_episode_reward(&mut self, episode_reward: f32) {
        self.episode_reward_history.add(episode_reward)
    }

    pub fn avg_episode_rewards(&self) -> f32 {
        let c = &self.episode_reward_history.buffer;
        assert!(!c.is_empty());
        c.iter().sum::<f32>() / c.len() as f32
    }

    pub fn get_many<const N: usize>(&self, indices: &[usize; N]) -> ReplaySample<N, S, A> {
        ReplaySample {
            state: self.state_history.get_many(&indices),
            state_next: self.state_next_history.get_many(&indices),
            reward: self.reward_history.get_many_as_val(&indices),
            action: self.action_history.get_many_as_val(&indices),
            done: self.done_history.get_many_as_val(&indices),
        }
    }
}

pub struct ReplaySample<'a, const N: usize, S, A> {
    pub state: [&'a S; N],
    pub state_next: [&'a S; N],
    pub reward: [f32; N],
    pub action: [A; N],
    pub done: [bool; N],
}