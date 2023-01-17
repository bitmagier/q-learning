use std::future::Future;
use std::sync::{Arc, RwLock};

use eframe::egui;
use egui::plot;
use egui::style;
use egui::widgets;
use crate::game_drawer::GameDrawer;

use crate::pong::game_api::{GameInput, GameState, PanelControl, Pong};
use crate::pong::pong_mechanics::PongMechanics;

pub struct PongApp {
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<GameState>>,
}

impl PongApp {
    pub fn new(cc: &eframe::CreationContext<'_>, game_input: Arc<RwLock<GameInput>>, game_state: Arc<RwLock<GameState>>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self {
            game_input,
            game_state,
        }
    }

    fn plot_game(&self, ui: &mut egui::Ui) {
        let game_state = self.game_state.read().unwrap().clone();
        let drawer = GameDrawer::new(game_state);
        let plot = plot::Plot::new("game_plot")
            .height(UI_HEIGHT)
            .width(UI_WIDTH)
            .view_aspect(1.0);
        // line, polygon, text, points, arrows, image, hline, vline, box_plot, bar_char
        plot.show(ui, |plot_ui|
            for p in drawer.polygons() {
                plot_ui.polygon(p)
            },
        );
    }
}

const UI_HEIGHT: f32 = 800.0;
const UI_WIDTH: f32 = 600.0;

impl eframe::App for PongApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Pong");
            // TODO draw game box

            ui.set_height(UI_HEIGHT);
            ui.set_width(UI_WIDTH);
            self.plot_game(ui);

            let control = if ctx.input().key_down(egui::Key::ArrowLeft) && !ctx.input().key_down(egui::Key::ArrowRight) {
                PanelControl::AccelerateLeft
            } else if ctx.input().key_down(egui::Key::ArrowRight) && !ctx.input().key_down(egui::Key::ArrowLeft) {
                PanelControl::AccelerateRight
            } else {
                PanelControl::None
            };
            let x = self.game_input.get_mut().unwrap();
            *x = GameInput { control };
        });
    }
}
