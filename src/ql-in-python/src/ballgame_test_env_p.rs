use pyo3::pyfunction;
use ql::test::ballgame_test_environment::BallGameTestEnvironment;

#[pyfunction]
pub fn create_ballgame_test_env_p() -> BallgameTestEnvP {
    BallgameTestEnvP(
        BallGameTestEnvironment::new()
    )
}
struct BallgameTestEnvP(BallGameTestEnvironment);

