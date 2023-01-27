use eframe::epaint::{CircleShape, Shape};
use egui::{Color32, Pos2, Rect, Rounding, Stroke, Vec2};
use egui::epaint::RectShape;
use crate::pong::Coordinate;

use crate::pong::mechanics::{Assert, Ball, Brick, GameState, MODEL_GRID_LEN_X, MODEL_GRID_LEN_Y, Panel};

pub struct GameDrawer {
    canvas_size: Vec2,
    game_state: GameState,
}

impl GameDrawer {
    pub fn new(canvas_size: Vec2, game_state: GameState) -> Self {
        Self {
            canvas_size,
            game_state,
        }
    }

    /// pos / MODEL_LEN = result / canvas_size
    /// => result = pos * canvas_size / MODEL_LEN
    fn scale(&self, pos: Coordinate) -> Pos2 {
        Pos2::new(
            pos.x * self.canvas_size.x / MODEL_GRID_LEN_X,
            pos.y * self.canvas_size.y / MODEL_GRID_LEN_Y,
        )
    }

    fn scale_x(&self, len_x: f32) -> f32 {
        len_x * self.canvas_size.x / MODEL_GRID_LEN_X
    }


    pub fn shapes(&self) -> Vec<egui::Shape> {
        let mut result = Vec::with_capacity(self.game_state.bricks.len() + 2);
        result.extend(self.bricks());
        result.push(self.ball());
        result.push(self.panel());
        result
    }

    fn bricks(&self) -> Vec<egui::Shape> {
        self.game_state.bricks.iter()
            .map(|b| self.draw_brick(b))
            .collect()
    }

    fn ball(&self) -> egui::Shape {
        self.draw_ball(&self.game_state.ball)
    }

    fn draw_ball(&self, ball: &Ball) -> Shape {
        ball.assert();
        CircleShape::stroke(
            self.scale(ball.center_pos),
            self.scale_x(ball.radius),
            Stroke::new(2.0, Color32::YELLOW),
        ).into()
    }


    fn panel(&self) -> egui::Shape {
        self.draw_panel(&self.game_state.panel)
    }

    fn draw_panel(&self, panel: &Panel) -> Shape {
        panel.assert();
        RectShape::filled(
            Rect::from_two_pos(
                self.scale(
                    Coordinate::new(
                        panel.center_pos_x - panel.size_x / 2.0,
                        panel.center_pos_y - panel.size_y / 2.0,
                    )),
                self.scale(Coordinate::new(
                    panel.center_pos_x + panel.size_x / 2.0,
                    panel.center_pos_y + panel.size_y / 2.0,
                )),
            ),
            Rounding::none(),
            Color32::WHITE,
        ).into()
    }


    fn draw_brick(&self, brick: &Brick) -> egui::Shape {
        RectShape::filled(
            Rect::from_two_pos(
                self.scale(brick.lower_left),
                self.scale(brick.upper_right),
            ),
            Rounding::none(),
            Color32::DARK_GRAY,
        ).into()
    }
}
