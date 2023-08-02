use egui::epaint::{CircleShape, RectShape};
use egui::{Color32, Pos2, Rect, Rounding, Shape, Stroke, Vec2};

use super::mechanics::{Assert, Ball, BreakoutMechanics, Brick, Panel, MODEL_GRID_LEN_X, MODEL_GRID_LEN_Y};

pub struct AppGameDrawer {
    canvas_size: Vec2,
    game_state: BreakoutMechanics,
}

impl AppGameDrawer {
    pub fn new(
        canvas_size: Vec2,
        game_state: BreakoutMechanics,
    ) -> Self {
        Self { canvas_size, game_state }
    }

    /// pos / MODEL_LEN = result / canvas_size
    /// => result = pos * canvas_size / MODEL_LEN
    fn scale(
        &self,
        pos: Pos2,
    ) -> Pos2 {
        Pos2::new(
            pos.x * self.canvas_size.x / MODEL_GRID_LEN_X,
            pos.y * self.canvas_size.y / MODEL_GRID_LEN_Y,
        )
    }

    fn scale_x(
        &self,
        len_x: f32,
    ) -> f32 {
        len_x * self.canvas_size.x / MODEL_GRID_LEN_X
    }

    pub fn shapes(&self) -> Vec<egui::Shape> {
        let mut result = Vec::with_capacity(self.game_state.bricks.len() + 2);
        result.extend(self.bricks());
        result.push(self.ball());
        result.push(self.panel());
        result
    }

    fn bricks(&self) -> Vec<egui::Shape> { self.game_state.bricks.iter().map(|b| self.draw_brick(b)).collect() }

    fn ball(&self) -> egui::Shape { self.draw_ball(&self.game_state.ball) }

    fn draw_ball(
        &self,
        ball: &Ball,
    ) -> Shape {
        ball.assert();
        CircleShape::stroke(
            self.scale(ball.shape.center),
            self.scale_x(ball.shape.radius),
            Stroke::new(2.0, Color32::YELLOW),
        )
        .into()
    }

    fn panel(&self) -> egui::Shape { self.draw_panel(&self.game_state.panel) }

    fn draw_panel(
        &self,
        panel: &Panel,
    ) -> Shape {
        panel.assert();
        RectShape::filled(
            Rect::from_two_pos(self.scale(panel.shape.min), self.scale(panel.shape.max)),
            Rounding::none(),
            Color32::WHITE,
        )
        .into()
    }

    fn draw_brick(
        &self,
        brick: &Brick,
    ) -> egui::Shape {
        RectShape::filled(
            Rect::from_two_pos(self.scale(brick.shape.min), self.scale(brick.shape.max)),
            Rounding::none(),
            Color32::DARK_GRAY,
        )
        .into()
    }
}
