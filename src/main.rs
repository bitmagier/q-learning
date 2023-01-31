// extern crate tensorflow;

use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use crate::app::PongApp;
use crate::pong::mechanics::{GameInput, GameState, PanelControl, PongMechanics, TIME_GRANULARITY};

pub mod pong;
mod app;
mod game_drawer;

fn main() {
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    let game_input = Arc::new(RwLock::new(GameInput::new()));
    let game_state = Arc::new(RwLock::new(GameState::default()));

    let m_game_input = Arc::clone(&game_input);
    let m_game_state = Arc::clone(&game_state);

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Pong", native_options, Box::new(|cc| {
        let egui_ctx = cc.egui_ctx.clone();
        let mechanics_join_handle = thread::spawn(move || mechanics_thread(m_game_input, m_game_state, egui_ctx));
        Box::new(PongApp::new(cc, game_input, game_state, mechanics_join_handle))
    }));
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

    let mut mechanics = PongMechanics::new();
    let mut next_step_time = Instant::now().add(TIME_GRANULARITY);
    let sleep_time_ms = TIME_GRANULARITY.div(5);
    loop {
        if Instant::now().ge(&next_step_time) {
            next_step_time = next_step_time.add(TIME_GRANULARITY);
            let input = read_input();
            if PanelControl::Exit == input.control {
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
