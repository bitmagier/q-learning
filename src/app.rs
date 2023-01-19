use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;

use eframe::egui;
use eframe::glow::Context;
use egui::{Frame, plot, Sense, Vec2};
use egui::epaint::PathShape;
use crate::game_drawer::GameDrawer;

use crate::pong::game_api::{GameInput, GameState, PanelControl};
use crate::pong::pong_mechanics::{BOARD_DIM_X, BOARD_DIM_Y};

pub struct PongApp<T> {
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<GameState>>,
    mechanics_join_handle: JoinHandle<T>
}

impl<T> PongApp<T> {
    pub fn new(cc: &eframe::CreationContext<'_>, game_input: Arc<RwLock<GameInput>>, game_state: Arc<RwLock<GameState>>, mechanics_join_handle: JoinHandle<T>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        Self {
            game_input,
            game_state,
            mechanics_join_handle
        }
    }

    fn ui_control(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let control = if ctx.input().key_down(egui::Key::ArrowLeft) && !ctx.input().key_down(egui::Key::ArrowRight) {
            PanelControl::AccelerateLeft
        } else if ctx.input().key_down(egui::Key::ArrowRight) && !ctx.input().key_down(egui::Key::ArrowLeft) {
            PanelControl::AccelerateRight
        } else if ctx.input().key_pressed(egui::Key::Escape) {
            PanelControl::Exit
        } else {
            PanelControl::None
        };
        *self.game_input.write().unwrap() = GameInput { control };
    }

    fn ui_content(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(UI_WIDTH, UI_HEIGHT), Sense::focusable_noninteractive());

        let drawer = GameDrawer::new(self.game_state.read().unwrap().clone());
        for shape in drawer.shapes() {
            painter.add(shape);
        }
        response
    }
}

const UI_HEIGHT: f32 = 800.0;
const UI_WIDTH: f32 = 600.0;

/// see https://github.com/emilk/egui/blob/master/crates/egui_demo_lib/src/demo/paint_bezier.rs

impl<T> eframe::App for PongApp<T> {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.mechanics_join_handle.is_finished() {
            frame.close()
        }
        egui::Window::new("pong")
            //.open(true)
            .vscroll(false)
            .hscroll(false)
            .resizable(false)
            .default_size([BOARD_DIM_X, BOARD_DIM_Y])
            .show(ctx, |ui| {

                self.ui_control(ctx, ui);

                Frame::canvas(ui.style()).show(ui, |ui| {
                    self.ui_content(ui)
                });

            });
    }
    
    fn on_exit(&mut self, _gl: Option<&Context>) {
        *self.game_input.write().unwrap() = GameInput { control: PanelControl::Exit };
    }
}
