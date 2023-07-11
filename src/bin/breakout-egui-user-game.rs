use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;

use eframe::glow;
use egui::{Context, Id, LayerId, Order, Painter, Vec2};

use q_learning_breakout::environment::breakout::app_game_drawer::AppGameDrawer;
use q_learning_breakout::environment::breakout::mechanics::*;
use q_learning_breakout::util::init_logging;

pub const FRAME_SIZE_X: usize = MODEL_GRID_LEN_X as usize;
pub const FRAME_SIZE_Y: usize = MODEL_GRID_LEN_Y as usize;

pub struct BreakoutApp {
    game_input: Arc<RwLock<GameInput>>,
    game_state: Arc<RwLock<BreakoutMechanics>>,
    mechanics_join_handle: JoinHandle<()>,
}

impl BreakoutApp {
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        game_input: Arc<RwLock<GameInput>>,
        game_state: Arc<RwLock<BreakoutMechanics>>,
        mechanics_join_handle: JoinHandle<()>,
    ) -> Self {
        Self {
            game_input,
            game_state,
            mechanics_join_handle,
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
        frame: &mut eframe::Frame,
    ) {
        if self.mechanics_join_handle.is_finished() {
            frame.close()
        }
        frame.set_window_size(Vec2::new(FRAME_SIZE_X as f32, FRAME_SIZE_Y as f32));

        let player_input = self.read_ui_control(ctx);
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
}

fn mechanics_thread(game_input: Arc<RwLock<GameInput>>, game_state: Arc<RwLock<BreakoutMechanics>>, egui_ctx: Context) {
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

    let mut mechanics = BreakoutMechanics::new();
    let mut next_step_time = Instant::now().add(TIME_GRANULARITY);
    let sleep_time_ms = TIME_GRANULARITY.div(5);
    loop {
        if Instant::now().ge(&next_step_time) {
            next_step_time = next_step_time.add(TIME_GRANULARITY);
            let input = read_input();
            if input.exit {
                break;
            } else {
                mechanics.time_step(input);
                if mechanics.finished {
                    log::info!("score: {:?}", &mechanics.score);
                    break;
                }
                write_game_state(mechanics.clone());
            }
            egui_ctx.request_repaint();
        }
        thread::sleep(sleep_time_ms);
    }
}

fn breakout_user_game() -> eframe::Result<()> {
    let game_input = Arc::new(RwLock::new(GameInput::none()));
    let game_state = Arc::new(RwLock::new(BreakoutMechanics::new()));

    let m_game_input = Arc::clone(&game_input);
    let m_game_state = Arc::clone(&game_state);

    let mut native_options = eframe::NativeOptions::default();
    native_options.default_theme = eframe::Theme::Dark;
    eframe::run_native("Breakout", native_options, Box::new(|cc| {
        let egui_ctx = cc.egui_ctx.clone();
        let mechanics_join_handle = thread::spawn(move || mechanics_thread(m_game_input, m_game_state, egui_ctx));
        Box::new(BreakoutApp::new(cc, game_input, game_state, mechanics_join_handle))
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    breakout_user_game()?;
    Ok(())
}

