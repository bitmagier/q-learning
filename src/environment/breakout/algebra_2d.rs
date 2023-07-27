use egui::{Pos2, Vec2};
use parry2d::na::{Isometry2, Vector2};
use parry2d::query;
use parry2d::query::Contact;
use parry2d::shape::{Ball, Cuboid};

use crate::environment::breakout::mechanics::CONTACT_PREDICTION;

/// Axis-aligned Bounding Box
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AaBB {
    pub min: Pos2,
    pub max: Pos2,
}

impl AaBB {
    pub fn center(&self) -> Pos2 {
        Pos2::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
        )
    }
}

impl AaBB {
    pub fn translate(&self, value: Vec2) -> Self {
        AaBB {
            min: self.min + value,
            max: self.max + value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    pub center: Pos2,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactSurface {
    pub way: f32,
    pub approximation: f32,
    // after way
    pub surface_normal: Vec2,
}

/// r = v - 2 (v ⋅ n) n
pub fn reflected_vector(v: Vec2, surface_normal: Vec2) -> Vec2 {
    v - 2.0 * v.dot(surface_normal) * surface_normal
}

/// for normalized vectors: cos(angle) = v1 ⋅ v2
pub fn vector_angle(v1: Vec2, v2: Vec2) -> f32 {
    v1.normalized().dot(v2.normalized()).acos()
}

pub fn contact_test_circle_aabb(circle: &Circle, aabb: &AaBB) -> Option<Contact> {
    let aabb_center = aabb.center();
    query::contact(
        &Isometry2::translation(circle.center.x, circle.center.y),
        &Ball::new(circle.radius),
        &Isometry2::translation(aabb_center.x, aabb_center.y),
        &Cuboid::new(Vector2::new(
            (aabb.max.x - aabb.min.x) / 2.0,
            (aabb.max.y - aabb.min.y) / 2.0,
        )),
        CONTACT_PREDICTION,
    )
        .expect("contact calculation failed")
}
