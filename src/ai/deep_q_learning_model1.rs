use lazy_static::lazy_static;

use crate::ai::result::AiError;

lazy_static!(
    static ref MODEL_ACTIONS: Vec<(ModelAction, u8)> = vec![
        (ModelAction::None, 0_u8),
        (ModelAction::AccelerateLeft, 1),
        (ModelAction::AccelerateRight, 2)
    ];
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelAction {
    None,
    AccelerateLeft,
    AccelerateRight,
}

impl TryFrom<u8> for ModelAction {
    type Error = AiError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match MODEL_ACTIONS.iter()
            .find(|(_, c)| *c == value) {
            None => Err(format!("'{}' does not correspond to a ModelAction", value)),
            Some(e) => Ok(e.0)
        }
    }
}

impl From<ModelAction> for u8 {
    fn from(value: ModelAction) -> Self {
        match MODEL_ACTIONS.iter()
            .find(|(e, _)| *e == value) {
            None => panic!("missing action in MODEL_ACTIONS"),
            Some(e) => e.1
        }
    }
}

struct ModelState {

}
impl ModelState {

    // TODO pub fn absorb_next_fame(&mut self)
}

struct DeepQLearningOnModel1 {}


pub fn _learn_until_fit() {
    todo!()
}

