use crate::app::{ExternalGameController, FRAME_SIZE};
use crate::breakout::mechanics::GameInput;


// TODO asynchronously run AI model (playing or learning mode)
pub struct GameAiAdapter {
}
impl ExternalGameController for GameAiAdapter {
    fn show_frame(&mut self, frame: [u8; FRAME_SIZE]) {
        todo!()
    }

    fn read_input(&mut self) -> GameInput {
        todo!()
    }
}

