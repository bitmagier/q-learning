use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use egui::{Context, Id, LayerId, Order, Painter, Vec2};


use crate::game_drawer::GameDrawer;
use crate::breakout::mechanics::{GameInput, GameState, PanelControl};

pub struct BreakoutApp<T> {
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<GameState>>,
    mechanics_join_handle: JoinHandle<T>,
}

impl<T> BreakoutApp<T> {
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        game_input: Arc<RwLock<GameInput>>,
        game_state: Arc<RwLock<GameState>>,
        mechanics_join_handle: JoinHandle<T>,
    ) -> Self {
        Self {
            game_input,
            game_state,
            mechanics_join_handle,
        }
    }

    fn ui_control(&mut self, ctx: &egui::Context) {
        let control = if ctx.input(|i| i.key_down(egui::Key::ArrowLeft) && !i.key_down(egui::Key::ArrowRight)) {
            PanelControl::AccelerateLeft
        } else if ctx.input(|i|i.key_down(egui::Key::ArrowRight) && !i.key_down(egui::Key::ArrowLeft)) {
            PanelControl::AccelerateRight
        } else {
            PanelControl::None
        };
        let exit = ctx.input(|i| i.key_down(egui::Key::Escape));

        self.write_game_input(GameInput { control, exit });
    }

    fn game_content(&self, painter: &Painter) {
        let paint_offset = painter.clip_rect().min;
        let canvas_size = painter.clip_rect().size();

        let game_state = self.read_game_state();
        let drawer = GameDrawer::new(canvas_size, game_state);
        for mut shape in drawer.shapes() {
            shape.translate(paint_offset.to_vec2());
            painter.add(shape);
        }
    }

    fn read_game_state(&self) -> GameState {
        let read_handle = self.game_state.read().unwrap();
        let game_state = read_handle.clone();
        drop(read_handle);
        game_state
    }

    fn write_game_input(&self, game_input: GameInput) {
        let mut write_handle = self.game_input.write().unwrap();
        *write_handle = game_input;
        drop(write_handle);
    }
}

/// see https://github.com/emilk/egui/blob/master/crates/egui_demo_lib/src/demo/paint_bezier.rs

pub const FRAME_SIZE_X: f32 = 600.0;
pub const FRAME_SIZE_Y: f32 = 800.0;

impl<T> eframe::App for BreakoutApp<T> {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        if self.mechanics_join_handle.is_finished() {
            frame.close()
        }
        frame.set_window_size(Vec2::new(FRAME_SIZE_X, FRAME_SIZE_Y));
        self.ui_control(ctx);
        let game_painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("game")));
        self.game_content(&game_painter);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        *self.game_input.write().unwrap() = GameInput { control: PanelControl::None, exit: true };
    }
}
