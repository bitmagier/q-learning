use image::{ImageBuffer, imageops, Rgb};

use crate::ai::realtime_ai_player::RealtimeAiPlayer;
use crate::app::ExternalGameController;
use crate::breakout::mechanics::GameInput;

use threadpool::ThreadPool;

// TODO asynchronously run AI model (playing or learning mode)
pub struct GameAiAdapter {
    player: RealtimeAiPlayer,
    next_action: Option<GameInput>
}
impl GameAiAdapter {
    pub fn new() -> Self {
        todo!()
    }
}
impl ExternalGameController for GameAiAdapter {
    fn show_frame(&mut self, frame: ImageBuffer<Rgb<u8>, Vec<u8>>) {


        let game_input = self.player.watch_next_frame(
            imageops::grayscale(&frame)
        );

        // TODO run tensorflow model and fill `next_action`
        todo!()
    }

    fn read_input(&mut self) -> GameInput {
        if let Some(action) = self.next_action {
            self.next_action = None;
            action
        } else {
            GameInput::none()
        }
    }
}

