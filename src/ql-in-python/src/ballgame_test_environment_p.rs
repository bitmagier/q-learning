use pyo3::{pyclass, pyfunction, PyResult};

use ql::prelude::Environment;
use ql::test::ballgame_test_environment::{BallGameAction, BallGameState, BallGameTestEnvironment};

#[pyfunction]
pub fn create_ballgame_test_environment() -> PyResult<BallGameTestEnvironmentP> {
    Ok(BallGameTestEnvironmentP(BallGameTestEnvironment::new()))
}

#[pyclass]
pub struct BallGameTestEnvironmentP(BallGameTestEnvironment);

impl BallGameTestEnvironmentP {
    pub fn reset(&mut self) {
        self.0.reset();
    }

    pub fn state(&self) -> PyResult<BallGameState> {
        Ok(self.0.state().clone())
    }

    pub fn step(&mut self, action: BallGameActionP) -> PyResult<(BallGameStateP, f32, bool)> {
        match self.0.step(action.0) {
            (state_ref, reward, done) => Ok((BallGameStateP(state_ref.clone()), reward, done))
        }
    }

    pub fn episode_reward_goal_mean(&self) -> f32 {
        self.0.episode_reward_goal_mean()
    }
}

#[pyclass]
pub struct BallGameStateP(BallGameState);

#[pyclass]
pub struct BallGameActionP(BallGameAction);