use std::collections::VecDeque;

struct Buffer<T> {
    max_buffer_len: usize,
    buffer: VecDeque<T>,
}

impl<T> Buffer<T> {
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
        if self.buffer.len() >= self.max_buffer_len {
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

impl<T: Copy> Buffer<T> {
    /// returns a slice of values of `slice` at the specified `indices`
    pub fn get_many_as_val<const N: usize>(&self, indices: &[usize; N]) -> [T; N] {
        debug_assert!(!indices.iter().any(|&e| e >= self.buffer.len()));
        indices.map(|i| self.buffer[i])
    }
}

/// Experience replay buffers
pub struct ReplayBuffer<S, A>
where A: Copy
{
    action_history: Buffer<A>,
    state_history: Buffer<S>,
    state_next_history: Buffer<S>,
    reward_history: Buffer<f32>,
    done_history: Buffer<bool>,
    episode_reward_history: Buffer<f32>,
}

impl<S, A> ReplayBuffer<S, A>
where A: Copy
{
    pub fn new(step_buffer_len: usize, episode_reward_buffer_len: usize) -> Self {
        Self {
            action_history: Buffer::new(step_buffer_len),
            state_history: Buffer::new(step_buffer_len),
            state_next_history: Buffer::new(step_buffer_len),
            reward_history: Buffer::new(step_buffer_len),
            done_history: Buffer::new(step_buffer_len),
            episode_reward_history: Buffer::new(episode_reward_buffer_len),
        }
    }

    pub fn len(&self) -> usize {
        self.done_history.len()
    }

    pub fn add_step_items(&mut self, action: A, state: S, state_next: S, reward: f32, done: bool) {
        self.action_history.add(action);
        self.state_history.add(state);
        self.state_next_history.add(state_next);
        self.reward_history.add(reward);
        self.done_history.add(done);
    }

    pub fn add_episode_reward(&mut self, episode_reward: f32) {
        self.episode_reward_history.add(episode_reward)
    }

    pub fn avg_episode_rewards(&self) -> Option<f32> {
        if self.episode_reward_history.len() < self.episode_reward_history.max_buffer_len {
            None
        } else {
            let c = &self.episode_reward_history.buffer;
            let avg = c.iter().sum::<f32>() / c.len() as f32;
            Some(avg)
        }
    }

    pub fn get_many<const N: usize>(&self, indices: &[usize; N]) -> BufferSample<N, S, A> {
        BufferSample {
            state: self.state_history.get_many(&indices),
            state_next: self.state_next_history.get_many(&indices),
            reward: self.reward_history.get_many_as_val(&indices),
            action: self.action_history.get_many_as_val(&indices),
            done: self.done_history.get_many_as_val(&indices),
        }
    }
}

pub struct BufferSample<'a, const N: usize, S, A> {
    pub state: [&'a S; N],
    pub state_next: [&'a S; N],
    pub reward: [f32; N],
    pub action: [A; N],
    pub done: [bool; N],
}