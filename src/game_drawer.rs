use egui::{Color32, Pos2, Stroke};
use egui::epaint::PathShape;

use crate::pong::game_api::{Brick, GameState};

pub struct GameDrawer {
    game_state: GameState,
}

impl GameDrawer {
    pub fn new(game_state: GameState) -> Self {
        Self {
            game_state
        }
    }

    pub fn shapes(&self) -> Vec<egui::Shape> {
        let mut result = Vec::with_capacity(self.game_state.bricks.len() + 2);
        result.extend(self.bricks());
        result.push(self.ball());
        result.push(self.panel());
        result
    }

    fn bricks(&self) -> Vec<egui::Shape> {
        todo!();
    }
    fn ball(&self) -> egui::Shape {
        todo!()
    }
    fn panel(&self) -> egui::Shape {
        todo!()
    }
}

fn draw_brick(brick: &Brick) -> impl Into<egui::Shape> {
    PathShape::convex_polygon(
        vec![
            Pos2::new(brick.lower_left.x, brick.lower_left.y),
            Pos2::new(brick.lower_left.x, brick.upper_right.y),
            Pos2::new(brick.upper_right.x, brick.upper_right.y),
            Pos2::new(brick.upper_right.x, brick.lower_left.y)
        ],
        Color32::DARK_GRAY,
        Stroke::new(1.0, Color32::LIGHT_GRAY)
    )
}
