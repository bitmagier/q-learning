use std::future::Future;
use std::sync::Arc;

use egui::Key;
use egui::Key::ArrowLeft;

use eframe::egui;

use crate::pong::game_api::{GameState, GameInput, Pong, PanelControl};
use crate::pong::pong_mechanics::PongMechanics;

pub struct PongApp {
    game_input: Arc<GameInput>
}

impl PongApp {
    pub fn new(cc: &eframe::CreationContext<'_>, game_input: Arc<GameInput>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self {
            game_input
        }
    }
}

impl eframe::App for PongApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Pong");
            // TODO draw game box
            self.game_input.control = if ctx.input().key_down(Key::ArrowLeft) && !ctx.input().key_down(Key::ArrowRight) {
                PanelControl::AccelerateLeft
            } else if ctx.input().key_down(Key::ArrowRight) && !ctx.input().key_down(ArrowLeft) {
                PanelControl::AccelerateRight
            } else {
                PanelControl::None
            };
        });
    }
}
