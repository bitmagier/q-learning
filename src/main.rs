#![feature(drain_filter)]
extern crate tensorflow;

use std::fmt::{Display, Formatter};
use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use crate::app::BreakoutApp;
use crate::breakout::mechanics::{BreakoutMechanics, GameInput, GameState, TIME_GRANULARITY};
use crate::game_ai_adapter::GameAiAdapter;

pub mod breakout;
mod app;
mod game_drawer;
mod ai;
mod game_ai_adapter;

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
enum GameMode {
    User,
    AiLearning,
    AiPlaying,
}

impl Display for GameMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            match self {
                GameMode::User => "User",
                GameMode::AiLearning => "AiLearning",
                GameMode::AiPlaying => "AiPlaying"
            })
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(value_enum, default_value_t = GameMode::User)]
    mode: GameMode,
}

fn main() -> eframe::Result<()> {
    init_logging();

    let args: Args = Args::parse();

    let game_input = Arc::new(RwLock::new(GameInput::new()));
    let game_state = Arc::new(RwLock::new(GameState::default()));

    let m_game_input = Arc::clone(&game_input);
    let m_game_state = Arc::clone(&game_state);

    let ai_game_controller: Option<GameAiAdapter> = match args.mode {
        GameMode::User => None,
        GameMode::AiLearning | GameMode::AiPlaying => Some(GameAiAdapter {})
    };

    let mut native_options = eframe::NativeOptions::default();
    native_options.default_theme = eframe::Theme::Dark;
    eframe::run_native("Breakout", native_options, Box::new(|cc| {
        let egui_ctx = cc.egui_ctx.clone();
        let mechanics_join_handle = thread::spawn(move || mechanics_thread(m_game_input, m_game_state, egui_ctx));
        Box::new(BreakoutApp::new(cc, game_input, game_state, mechanics_join_handle, ai_game_controller))
    }))
}

fn init_logging() {
    if cfg!(debug_assertions) {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }
}

fn mechanics_thread(game_input: Arc<RwLock<GameInput>>, game_state: Arc<RwLock<GameState>>, egui_ctx: egui::Context) {
    let read_input = || -> GameInput {
        let read_handle = game_input.read().unwrap();
        let input = read_handle.clone();
        drop(read_handle);
        input
    };

    let write_game_state = |state| {
        let mut write_handle = game_state.write().unwrap();
        *write_handle = state;
        drop(write_handle);
    };

    let mut mechanics = BreakoutMechanics::new();
    let mut next_step_time = Instant::now().add(TIME_GRANULARITY);
    let sleep_time_ms = TIME_GRANULARITY.div(5);
    loop {
        if Instant::now().ge(&next_step_time) {
            next_step_time = next_step_time.add(TIME_GRANULARITY);
            let input = read_input();
            if input.exit {
                break;
            } else {
                let state = mechanics.time_step(input);
                if state.finished {
                    log::debug!("{:?}", state.clone().game_result);
                    break;
                }
                write_game_state(state);
            }
            egui_ctx.request_repaint();
        }
        thread::sleep(sleep_time_ms);
    }
}
