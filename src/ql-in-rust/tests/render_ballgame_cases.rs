use anyhow::Result;

use common::consts::BATCH_SIZE;
use ql::prelude::{DebugVisualizer, Environment, QlError};
use ql::test::ballgame_test_environment::{BallGameState, BallGameTestEnvironment};
use ql::util::dbscan::cluster_analysis;
use ql_in_rust::ml_model::model::DeepQLearningModel;
use ql_in_rust::ml_model::tensorflow_python::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_TRAINED_PATH, QLearningTensorflowModel};

mod common;

#[test]
fn check_cases() -> Result<()> {
    use std::io::Write;
    fn simulate_outcome(
        env: BallGameTestEnvironment,
        model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>,
    ) -> f32 {
        let mut env = env;
        loop {
            let action = model.predict_action(env.state());
            let (_, reward, done) = env.step(action);
            if done {
                break reward;
            }
        }
    }

    let model = load_model()?;
    let mut rewards = vec![];
    for state in BallGameState::all_possible_initial_states() {
        let env = BallGameTestEnvironment::from(state);
        let reward = simulate_outcome(env, &model);
        rewards.push(reward);
    }
    let stdout = std::io::stdout();
    writeln!(&stdout, "All cases final reward: {}", cluster_analysis(&rewards, 0.3, 3))?;

    Ok(())
}

#[test]
fn render_a_successful_case() -> Result<()> {
    println!("rendering a walk through a successful case:");
    render_case(|model| find_successful_case(model))
}

// In a successful learning scenario this test will fail - but it's here in case we need it for debugging a failed learning algorithm
// #[test]
// fn render_unsuccessful_case() -> Result<()> {
//     println!("rendering a walk through an unsuccessful case:");
//     render_case(|model| find_unsuccessful_case(model))
// }

fn load_model() -> Result<QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>> {
    let model = QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load_model(&QL_MODEL_BALLGAME_3x3x4_5_512_TRAINED_PATH)?;
    Ok(model)
}

fn render_case(
    case_initial_state_select_fn: fn(&QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameState>
) -> Result<()> {
    let model = load_model()?;

    let initial_state = case_initial_state_select_fn(&model)?;
    let mut env = BallGameTestEnvironment::from(initial_state);
    render(env.state())?;

    loop {
        let predicted_action = model.predict_action(env.state());
        let (_, _, done) = env.step(predicted_action);

        render(env.state())?;
        if done {
            break;
        }
    }

    Ok(())
}

fn find_successful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameState> {
    for initial_state in BallGameState::all_possible_initial_states() {
        let mut env = BallGameTestEnvironment::from(initial_state.clone());
        loop {
            let action = model.predict_action(env.state());
            let (_, r, done) = env.step(action);
            if done {
                if r >= env.episode_reward_goal_mean() {
                    return Ok(initial_state);
                } else {
                    break;
                }
            }
        }
    }
    Err(QlError::from("could not find successful case"))?
}

#[allow(dead_code)]
fn find_unsuccessful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameState> {
    for initial_state in BallGameState::all_possible_initial_states() {
        let mut env = BallGameTestEnvironment::from(initial_state.clone());
        loop {
            let action = model.predict_action(env.state());
            let (_, r, done) = env.step(action);
            if done {
                if r < env.episode_reward_goal_mean() {
                    return Ok(initial_state);
                } else {
                    break;
                }
            }
        }
    }
    Err(QlError::from("could not find unsuccessful case"))?
}

fn render(state: &BallGameState) -> Result<()> {
    use std::io::Write;
    let console = state.render_to_console();
    let stdout = std::io::stdout();
    writeln!(&stdout, "step {}:", state.steps())?;
    console.draw();
    writeln!(&stdout, "\n-----")?;
    Ok(())
}
