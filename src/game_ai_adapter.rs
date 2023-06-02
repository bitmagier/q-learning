use crate::app::{ExternalGameController, FRAME_SIZE};
use crate::breakout::mechanics::{GameInput, PanelControl};
use crate::ai::realtime_ai_player::{GrayFrame, RealtimeAiPlayer};

// TODO asynchronously run AI model (playing or learning mode)
pub struct GameAiAdapter {
    player: RealtimeAiPlayer,
    action_buffer: Option<GameInput>
}
impl GameAiAdapter {
    pub fn new() -> Self {
        todo!()
    }
}
impl ExternalGameController for GameAiAdapter {
    fn show_frame(&mut self, frame: [u8; FRAME_SIZE]) {
        self.player.watch_next_frame(
            GrayFrame::from_rgb(frame)
        );

        todo!()
    }

    fn read_input(&mut self) -> GameInput {
        if let Some(action) = self.action_buffer {
            self.action_buffer = None;
            action
        } else {
            GameInput::none()
        }
    }
}

