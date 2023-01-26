// extern crate tensorflow;

extern crate core;

use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Instant};

use crate::app::PongApp;
use crate::pong::game::{GameInput, GameState, PanelControl, Pong};
use crate::pong::pong_mechanics::{PongMechanics, MODEL_TIME_PORTION};

pub mod pong;
mod app;
mod game_drawer;

fn main() {
    let game_input = Arc::new(RwLock::new(GameInput::new()));
    let game_state = Arc::new(RwLock::new(GameState::default()));

    let m_game_input = Arc::clone(&game_input);
    let m_game_state = Arc::clone(&game_state);
    let mechanics_join_handle = thread::spawn(move || mechanics_thread(m_game_input, m_game_state));

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Pong", native_options, Box::new(|cc|
        Box::new(PongApp::new(cc, game_input, game_state, mechanics_join_handle))));
}

fn mechanics_thread(game_input: Arc<RwLock<GameInput>>, game_state: Arc<RwLock<GameState>>) {
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

    let mut mechanics = PongMechanics::new();
    let mut next_step_time = Instant::now().add(MODEL_TIME_PORTION);
    let sleep_time_ms = MODEL_TIME_PORTION.div(5);
    loop {
        if Instant::now().ge(&next_step_time) {
            next_step_time = next_step_time.add(MODEL_TIME_PORTION);
            let input = read_input();
            if PanelControl::Exit == input.control {
                break
            } else {
                let state = mechanics.time_step(input);
                write_game_state(state);
            }
        }
        thread::sleep(sleep_time_ms);
    }
}
