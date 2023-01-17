// extern crate tensorflow;

use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::PongApp;
use crate::pong::game_api::{GameInput, GameState, PanelControl, Pong, TIME_GRANULARITY};
use crate::pong::pong_mechanics::PongMechanics;

pub mod pong;
mod app;
mod game_drawer;

fn main() {
    let game_input = RwLock::new(GameInput::new());
    let game_state = RwLock::new(GameState::default());

    let mechanics_thread = thread::spawn(|| {
        let mut mechanics = PongMechanics::new();
        let mut next_step_time = Instant::now().add(TIME_GRANULARITY);
        let sleep_time_ms = TIME_GRANULARITY.div(5);
        loop {
            if Instant::now().ge(&next_step_time) {
                next_step_time = next_step_time.add(TIME_GRANULARITY);
                *m_game_state.write().unwrap() = mechanics.time_step(m_game_input.read().unwrap().clone());
            }
            thread::sleep(sleep_time_ms);
        }
    });

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Pong", native_options, Box::new(|cc| Box::new(PongApp::new(cc, game_input, game_state))));
    mechanics_thread.join().unwrap();
}
