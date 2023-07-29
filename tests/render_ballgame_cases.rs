use anyhow::Result;

use common::{BATCH_SIZE, CHECKPOINT_FILE_BASE};
use q_learning_breakout::environment::ballgame_test_environment::{BallGameState, BallGameTestEnvironment};
use q_learning_breakout::ql::model::tensorflow::q_learning_model::{QL_MODEL_BALLGAME_3x3x4_5_512_PATH, QLearningTensorflowModel};
use q_learning_breakout::ql::prelude::{DebugVisualizer, Environment, DeepQLearningModel, QlError};

mod common;

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


fn render_case(case_select_fn: fn(&QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment>) -> Result<()> {
    if !CHECKPOINT_FILE_BASE.with_extension("index").exists() {
        return Err(QlError::from("Model checkpoint file does not yet exist. You might consider running testcase `test_learn_ballgame_until_mastered` first"))?
    }
    
    let model = QLearningTensorflowModel::<BallGameTestEnvironment, BATCH_SIZE>::load(&QL_MODEL_BALLGAME_3x3x4_5_512_PATH);
    model.read_checkpoint(CHECKPOINT_FILE_BASE.to_str().unwrap());

    let mut env = case_select_fn(&model)?;
    // let mut env: BallGameTestEnvironment = find_successful_case(
    //     &model
    // )?;
    render(env.state())?;

    loop {
        let predicted_action = model.predict_action(env.state());
        let (_, _, done) = env.step(predicted_action);

        render(env.state())?;
        if done {
            break
        }
    }

    Ok(())
}

fn find_successful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment> {
    let max_episodes = 10_000;
    let mut episode: usize = 0;
    loop {
        episode += 1;
        if episode > max_episodes {
            return Err(QlError::from("could not find successful case"))?
        }
        let candidate = BallGameTestEnvironment::default();
        let mut env = candidate.clone();
        let mut reward_sum = 0.0;
        loop {
            let action = model.predict_action(env.state());
            let (_,r,done) = env.step(action);
            reward_sum += r;
            if done && reward_sum >= env.episode_reward_goal_mean() {
                return Ok(candidate)
            }
            if done {
                break;
            }
        }
    }
}

fn find_unsuccessful_case(model: &QLearningTensorflowModel<BallGameTestEnvironment, BATCH_SIZE>) -> Result<BallGameTestEnvironment> {
    let max_episodes = 10_000;
    let mut episode: usize = 0;
    loop {
        episode += 1;
        if episode > max_episodes {
            return Err(QlError::from("could not find unsuccessful case"))?
        }
        let candidate = BallGameTestEnvironment::default();
        let mut env = candidate.clone();
        let mut reward_sum = 0.0;
        loop {
            let action = model.predict_action(env.state());
            let (_,r,done) = env.step(action);
            reward_sum += r;
            if done {
                if reward_sum >= env.episode_reward_goal_mean() {
                    break
                } else {
                    return Ok(candidate)
                }
            }
        }
    }
}

fn render(state: &BallGameState) -> Result<()>{
    use std::io::Write;
    let console = state.render_to_console();
    let stdout = std::io::stdout();
    writeln!(&stdout, "step {}:", state.steps())?;
    console.draw();
    writeln!(&stdout, "\n-----")?;
    Ok(())
}