// extern crate tensorflow;

extern crate core;

use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Instant};

use crate::app::PongApp;
use crate::pong::game::{GameInput, GameState, Pong};
use crate::pong::pong_mechanics::{PongMechanics, MODEL_TIME_PORTION};

pub mod pong;
mod app;
mod game_drawer;

fn main() {
    let game_input = Arc::new(RwLock::new(GameInput::new()));
    let game_state = Arc::new(RwLock::new(GameState::default()));

    let m_game_input = Arc::clone(&game_input);
    let m_game_state = Arc::clone(&game_state);
    let mechanics_join_handle = thread::spawn(move || {
        let mut mechanics = PongMechanics::new();
        let mut next_step_time = Instant::now().add(MODEL_TIME_PORTION);
        let sleep_time_ms = MODEL_TIME_PORTION.div(5);
        loop {
            if Instant::now().ge(&next_step_time) {
                next_step_time = next_step_time.add(MODEL_TIME_PORTION);
                let input = m_game_input.read().unwrap().clone();
                let state = mechanics.time_step(input);
                if state.finished {
                    break;
                }
                *m_game_state.write().unwrap() = state;
            }
            thread::sleep(sleep_time_ms);
        }
    });

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Pong", native_options, Box::new(|cc| Box::new(PongApp::new(cc, game_input, game_state, mechanics_join_handle))));
}
