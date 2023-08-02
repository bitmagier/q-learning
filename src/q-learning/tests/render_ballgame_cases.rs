use anyhow::Result;

use common::{BATCH_SIZE, CHECKPOINT_FILE_BASE};
use q_learning_breakout::environment::ballgame_test_environment::{BallGameState, BallGameTestEnvironment};
use q_learning_breakout::ql::model::tensorflow_python::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};
use q_learning_breakout::ql::prelude::{DebugVisualizer, DeepQLearningModel, Environment, QlError};
use q_learning_breakout::util::dbscan::cluster_analysis;

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

#[test]
fn render_unsuccessful_case() -> Result<()> {
    println!("rendering a walk through an unsuccessful case:");
    render_case(|model| find_unsuccessful_case(model))
}

fn load_model() -> Result<QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>> {
    // if !CHECKPOINT_FILE_BASE.with_extension("index").exists() {
    //     return Err(QlError::from(
    //         "Model checkpoint file does not yet exist. You might consider running testcase 'learn_ballgame' first",
    //     ))?;
    // }

    let mut model = QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load_model(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH)?;
    //model.read_checkpoint(CHECKPOINT_FILE_BASE.to_str().unwrap())?;
    model.load_graph(&CHECKPOINT_FILE_BASE)?;
    // TODO some kind of init missing - because it does not seem to be the trained model here
    Ok(model)
}

fn render_case(
    case_select_fn: fn(&QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment>
) -> Result<()> {
    let model = load_model()?;

    let mut env = case_select_fn(&model)?;
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

fn find_successful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment> {
    let mut episodes_left = 10_000;
    let mut candidate = BallGameTestEnvironment::default();

    loop {
        episodes_left -= 1;
        if episodes_left <= 0 {
            return Err(QlError::from("could not find successful case"))?;
        }

        let mut env = candidate.clone();
        loop {
            let action = model.predict_action(env.state());
            let (_, r, done) = env.step(action);
            if done && r >= env.episode_reward_goal_mean() {
                return Ok(candidate);
            }
            if done {
                break;
            }
        }
        candidate.reset();
    }
}

fn find_unsuccessful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment> {
    let mut episodes_left = 10_000;
    let mut candidate = BallGameTestEnvironment::default();

    loop {
        episodes_left -= 1;
        if episodes_left <= 0 {
            return Err(QlError::from("could not find unsuccessful case"))?;
        }

        let mut env = candidate.clone();
        loop {
            let action = model.predict_action(env.state());
            let (_, r, done) = env.step(action);
            if done {
                if r >= env.episode_reward_goal_mean() {
                    break;
                } else {
                    return Ok(candidate);
                }
            }
        }

        candidate.reset();
    }
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
