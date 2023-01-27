use std::ops::Add;

pub mod mechanics;

#[derive(Copy, Clone)]
pub struct Vector2d {
    pub x: f32,
    pub y: f32,
}

impl Vector2d {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// normalize to len = 1.0
    pub fn normalize(&mut self) {
        let len = (self.x.powi(2) + self.y.powi(2)).sqrt();
        if (1.0 - len).abs() > 0.001 {
            let factor = 1.0 / len;
            self.x = self.x * factor;
            self.y = self.y * factor;
        }
    }
}

impl std::ops::Mul<f32> for Vector2d {
    type Output = Vector2d;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector2d {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl From<(f32, f32)> for Vector2d {
    fn from(value: (f32, f32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<(isize, isize)> for Vector2d {
    fn from(value: (isize, isize)) -> Self {
        Self {
            x: value.0 as f32,
            y: value.1 as f32,
        }
    }
}

pub type Coordinate = Vector2d;

impl Add<Vector2d> for Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: Vector2d) -> Self::Output {
        Coordinate {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}


fn min_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::INFINITY, |a, b| a.min(b));
    assert!(r.is_finite());
    r
}

fn max_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::NEG_INFINITY, |a, b| a.max(b));
    assert!(r.is_finite());
    r
}
