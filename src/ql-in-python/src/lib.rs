mod ballgame_test_env_p;

use pyo3::prelude::*;
use ballgame_test_env_p::create_ballgame_test_env_p;

#[pymodule]
fn ql_in_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_ballgame_test_env_p, m)?)?;
    Ok(())
}