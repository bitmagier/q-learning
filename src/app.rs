use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use eframe::Frame;
use egui::{Context, Id, LayerId, Order, Painter, Vec2};
use eframe::glow;


use crate::game_drawer::GameDrawer;
use crate::breakout::mechanics::{GameInput, GameState, MODEL_GRID_LEN_X, MODEL_GRID_LEN_Y, PanelControl};

pub struct BreakoutApp<T> {
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<GameState>>,
    mechanics_join_handle: JoinHandle<T>,
    frame_receiver: Option<fn([u8; FRAME_SIZE])>
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
            frame_receiver: None
        }
    }

    // TODO use frame_receiver to fill ai::ModelState
    pub fn set_frame_receiver(&mut self, receiver: fn([u8; FRAME_SIZE])) {
        self.frame_receiver = Some(receiver);
    }

    fn ui_control(&mut self, ctx: &Context) {
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

pub const FRAME_SIZE_X: u32 = MODEL_GRID_LEN_X as u32;
pub const FRAME_SIZE_Y: u32 = MODEL_GRID_LEN_Y as u32;
pub const FRAME_SIZE: usize = (FRAME_SIZE_Y * FRAME_SIZE_Y * 3) as usize;

impl<T> eframe::App for BreakoutApp<T> {
    fn update(&mut self, ctx: &Context, frame: &mut Frame) {
        if self.mechanics_join_handle.is_finished() {
            frame.close()
        }
        frame.set_window_size(Vec2::new(FRAME_SIZE_X as f32, FRAME_SIZE_Y as f32));
        self.ui_control(ctx);
        let game_painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("game")));
        self.game_content(&game_painter);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        *self.game_input.write().unwrap() = GameInput { control: PanelControl::None, exit: true };
    }

    fn post_rendering(&mut self, window_size_px: [u32; 2], frame: &Frame) {
        assert_eq!(window_size_px[0], FRAME_SIZE_X);
        assert_eq!(window_size_px[1], FRAME_SIZE_Y);
        if let Some(frame_receiver) = self.frame_receiver {
            let gl = frame.gl().expect("need a GL context").clone();
            let painter = eframe::egui_glow::Painter::new(gl, "", None).expect("should be able to create glow painter");
            let frame = painter.read_screen_rgb(window_size_px);
            let frame: [u8; FRAME_SIZE] = frame.try_into()
                .unwrap_or_else(|v: Vec<u8>| panic!("vector of len {} should have length {}", v.len(), FRAME_SIZE));
            (frame_receiver)(frame)
        }
    }
}
