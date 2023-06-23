use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;

use eframe::Frame;
use eframe::glow;
use egui::{Context, Id, LayerId, Order, Painter, Vec2};
use image::{ImageBuffer, imageops, Rgb, RgbImage};
use image::imageops::FilterType;

use q_learning_breakout::breakout::mechanics::{BreakoutMechanics, GameInput, MODEL_GRID_LEN_X, MODEL_GRID_LEN_Y, PanelControl};
use q_learning_breakout::breakout::app_game_drawer::AppGameDrawer;

pub const FRAME_SIZE_X: usize = MODEL_GRID_LEN_X as usize;
pub const FRAME_SIZE_Y: usize = MODEL_GRID_LEN_Y as usize;

pub trait ExternalGameController {
    fn show_frame(&mut self, frame: ImageBuffer<Rgb<u8>, Vec<u8>>);
    fn read_input(&mut self) -> GameInput;
}

pub struct BreakoutApp {
    pixels_per_point: f32,
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<BreakoutMechanics>>,
    mechanics_join_handle: JoinHandle<()>,
    external_game_controller: Option<Box<dyn ExternalGameController>>,
}

impl BreakoutApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        game_input: Arc<RwLock<GameInput>>,
        game_state: Arc<RwLock<BreakoutMechanics>>,
        mechanics_join_handle: JoinHandle<()>,
        external_game_controller: Option<Box<dyn ExternalGameController>>,
    ) -> Self {
        Self {
            pixels_per_point: cc.integration_info.native_pixels_per_point.unwrap_or(1.0),
            game_input,
            game_state,
            mechanics_join_handle,
            external_game_controller,
        }
    }

    fn read_ui_control(
        &mut self,
        ctx: &Context,
    ) -> GameInput {
        let control = if ctx.input(|i| i.key_down(egui::Key::ArrowLeft) && !i.key_down(egui::Key::ArrowRight)) {
            PanelControl::AccelerateLeft
        } else if ctx.input(|i| i.key_down(egui::Key::ArrowRight) && !i.key_down(egui::Key::ArrowLeft)) {
            PanelControl::AccelerateRight
        } else {
            PanelControl::None
        };
        let exit = ctx.input(|i| i.key_down(egui::Key::Escape));
        GameInput { control, exit }
    }

    fn draw_game_content(&self, painter: &Painter) {
        let paint_offset = painter.clip_rect().min;
        let canvas_size = painter.clip_rect().size();

        let game_state = self.read_game_state();
        let drawer = AppGameDrawer::new(canvas_size, game_state);
        for mut shape in drawer.shapes() {
            shape.translate(paint_offset.to_vec2());
            painter.add(shape);
        }
    }

    fn read_game_state(&self) -> BreakoutMechanics {
        let read_handle = self.game_state.read().unwrap();
        let game_state = read_handle.clone();
        drop(read_handle);
        game_state
    }

    fn write_game_input(
        &self,
        game_input: GameInput,
    ) {
        let mut write_handle = self.game_input.write().unwrap();
        *write_handle = game_input;
        drop(write_handle);
    }
}

/// see https://github.com/emilk/egui/blob/master/crates/egui_demo_lib/src/demo/paint_bezier.rs

impl eframe::App for BreakoutApp {
    fn update(
        &mut self,
        ctx: &Context,
        frame: &mut Frame,
    ) {
        if self.mechanics_join_handle.is_finished() {
            frame.close()
        }
        frame.set_window_size(Vec2::new(FRAME_SIZE_X as f32, FRAME_SIZE_Y as f32));

        let player_input = if let Some(c) = &mut self.external_game_controller {
            c.read_input()
        } else {
            self.read_ui_control(ctx)
        };
        self.write_game_input(player_input);

        let game_painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("game")));
        self.draw_game_content(&game_painter);
    }

    fn on_exit(
        &mut self,
        _: Option<&glow::Context>,
    ) {
        *self.game_input.write().unwrap() = GameInput { control: PanelControl::None, exit: true };
    }

    fn post_rendering(
        &mut self,
        window_size_px: [u32; 2],
        frame: &Frame,
    ) {
        let display_scaling_factor = self.pixels_per_point;
        assert_eq!((window_size_px[0] as f32 / display_scaling_factor).round() as usize, FRAME_SIZE_X);
        assert_eq!((window_size_px[1] as f32 / display_scaling_factor).round() as usize, FRAME_SIZE_Y);

        if let Some(c) = &mut self.external_game_controller {
            let gl = frame.gl().expect("need a GL context").clone();
            let painter = eframe::egui_glow::Painter::new(gl, "", None).expect("should be able to create glow painter");
            let raw_frame = painter.read_screen_rgb(window_size_px);
            let normalized_frame: ImageBuffer<Rgb<u8>, Vec<u8>> = normalize_frame(raw_frame, window_size_px, display_scaling_factor);
            c.show_frame(normalized_frame)
        }
    }
}

fn normalize_frame(raw_frame: Vec<u8>, frame_size_px: [u32; 2], display_scale_factor: f32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let image = RgbImage::from_raw(frame_size_px[0], frame_size_px[1], raw_frame).expect("frame dimension should match");
    let scaling_factor = 1.0 / display_scale_factor;
    let new_width: u32 = (frame_size_px[0] as f32 * scaling_factor).round() as u32;
    let new_height: u32 = (frame_size_px[1] as f32 * scaling_factor).round() as u32;
    imageops::resize(&image, new_width, new_height, FilterType::Nearest)
}
