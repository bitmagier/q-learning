use egui::Color32;
use crate::app;
use crate::breakout::mechanics::{GameInput, PanelControl};

trait AiInterface {
    fn feed(&mut self, feed: StateVisual) -> AiFeedback;
}

struct StateVisual {
    screen: [[Color32; app::FRAME_SIZE_Y as usize]; app::FRAME_SIZE_X as usize]
}

struct AiFeedback {
    panel_control : PanelControl
}
