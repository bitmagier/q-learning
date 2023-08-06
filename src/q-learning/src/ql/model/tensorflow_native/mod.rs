//! Development starting point is Rust's native tensorflow binding.
//! So we work without any Keras convenience layer.
//!
//! Here is a nice example for how to build, train, save and load a tensorflow model from Rust:
//! https://github.com/nogibjj/assimilate-tensorflow/blob/2072ad867a7be2e134ca9ef46ccbd733793a9745/kick-tires-rust-tf/src/main.rs#L115
//!
//! But at the current development stage of tensorflow/rust it seems, that we would have a quite limited Tensorflow support only.
//! Using Python to define the model and (as it seems we are forced to do) training the model in python too seems much more powerful to me.
