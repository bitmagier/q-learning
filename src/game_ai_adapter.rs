use std::sync::mpsc;
use std::sync::mpsc::{Receiver, TryRecvError};
use image::{ImageBuffer, imageops, Rgb};

use crate::ai::realtime_ai_learner::RealtimeAiLearner;
use crate::app::ExternalGameController;
use crate::breakout::mechanics::{GameInput, PanelControl};

pub struct GameAiAdapter {
    player: RealtimeAiLearner,
    action_receiver: Receiver<GameInput>,
}

impl GameAiAdapter {
    pub fn new() -> Self {
        let (action_sender, action_receiver) = mpsc::channel();

        Self {
            player: RealtimeAiLearner::new(action_sender),
            action_receiver,
        }
    }
}

impl ExternalGameController for GameAiAdapter {
    fn show_frame(&mut self, frame: ImageBuffer<Rgb<u8>, Vec<u8>>) {
        self.player.watch_next_frame(imageops::grayscale(&frame))
    }

    fn read_input(&mut self) -> GameInput {
        match self.action_receiver.try_recv() {
            Ok(action) => action,
            Err(TryRecvError::Disconnected) => GameInput { control: PanelControl::None, exit: true },
            Err(TryRecvError::Empty) => GameInput::none()
        }
    }
}

