use egui::Color32;
use egui::plot::Polygon;

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

    pub fn polygons(&self) -> Vec<Polygon> {
        let mut result = Vec::with_capacity(self.game_state.bricks.len() + 2);

        for brick in &self.game_state.bricks {
            result.push(draw_brick(brick));
        }

        result
    }
}

fn draw_brick(brick: &Brick) -> Polygon {
    let mut points: Vec<[f64; 2]> = vec![];
    points.push([brick.lower_left.0, brick.lower_left.1]);
    points.push([brick.lower_left.0, brick.upper_right.1]);
    points.push([brick.upper_right.0, brick.upper_right.1]);
    points.push([brick.upper_right.0, brick.lower_left.1]);
    Polygon::new(points)
        .color(Color32::GRAY)
}
