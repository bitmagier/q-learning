use futures::channel::mpsc;
use futures::channel::mpsc::Receiver;
use image::{ImageBuffer, imageops, Rgb};

use crate::ai::realtime_ai_player::RealtimeAiPlayer;
use crate::app::ExternalGameController;
use crate::breakout::mechanics::{GameInput, PanelControl};

// TODO asynchronously run AI model (in learning mode / or in playing mode with a pre-trained model)
pub struct GameAiAdapter {
    player: RealtimeAiPlayer,
    action_receiver: Receiver<GameInput>,
}

impl GameAiAdapter {
    pub fn new() -> Self {
        let (action_sender, action_receiver) = mpsc::channel(1);
        Self {
            player: RealtimeAiPlayer::new(action_sender),
            action_receiver,
        }
    }
}

impl ExternalGameController for GameAiAdapter {
    fn show_frame(&mut self, frame: ImageBuffer<Rgb<u8>, Vec<u8>>) {
        self.player.watch_next_frame(imageops::grayscale(&frame))
    }

    fn read_input(&mut self) -> GameInput {
        match self.action_receiver.try_next() {
            Ok(Some(action)) => action,
            Ok(None) => GameInput { control: PanelControl::None, exit: true }, // AI player thread stopped unexpectedly
            Err(_) => GameInput::none()
        }
    }
}

