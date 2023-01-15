// extern crate tensorflow;

use std::sync::Arc;

use eframe::egui;

use crate::app::PongApp;
use crate::pong::game_api::{GameInput, PanelControl};

pub mod pong;
mod app;

fn main() {
    let native_options = eframe::NativeOptions::default();
    let game_input = Arc::new(GameInput::new());
    eframe::run_native("Pong", native_options, Box::new(|cc| Box::new(PongApp::new(cc, Arc::clone(&game_input)))));
}
