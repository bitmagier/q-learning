use std::f32::consts::FRAC_PI_2;
use std::ops::Range;
use std::time::Duration;

use egui::{Pos2, Vec2};
use itertools::Itertools;
use rand::Rng;

use crate::environment::breakout::algebra_2d::{AaBB, Circle, contact_test_circle_aabb, ContactSurface, reflected_vector, vector_angle};

/// TOP / LEFT corner is 0/0
pub const MODEL_GRID_LEN_X: f32 = 600.0;
pub const MODEL_GRID_LEN_Y: f32 = 600.0;

const CEILING_HEIGHT_Y: f32 = 0.0;

pub const SPACE_GRANULARITY: f32 = 0.001;
pub const TIME_GRANULARITY: Duration = Duration::from_millis(20);

const PANEL_LEN_X: f32 = 60.0;
const PANEL_LEN_Y: f32 = 10.0;
const PANEL_CENTER_POS_Y: f32 = MODEL_GRID_LEN_Y - 30.0;

const PANEL_MAX_SPEED_PER_SECOND: f32 = 160.0;

const PANEL_CONTROL_ACCEL_PER_SECOND: f32 = 20.0;
/// slow down if not accelerated
const PANEL_SLOW_DOWN_ACCEL_PER_SECOND: f32 = 7.0;

const BRICK_EDGE_LEN: f32 = 25.0;

const BRICKS_SETUP_SPACING: f32 = 2.0;
const BRICKS_SETUP_ROWS: usize = 3;
const BRICKS_SETUP_DISTANCE_LEFT_WALL: f32 = BALL_RADIUS * 3.0;
const BRICKS_SETUP_MIN_DISTANCE_RIGHT_WALL: f32 = BRICKS_SETUP_DISTANCE_LEFT_WALL;

const BRICKS_SETUP_FIRST_ROW_TOP_Y: f32 = 60.0;

const BALL_RADIUS: f32 = 10.0;
const BALL_SPEED_PER_SEC: f32 = 200.0;

// max object distance to detect a collision
pub(super) const CONTACT_PREDICTION: f32 = 0.8;
const CONTACT_PENETRATION_LIMIT: f32 = 0.0;


#[derive(Clone, Debug)]
pub struct BreakoutMechanics {
    /// x = 0 = left side; y = 0 = bottom
    pub bricks: Vec<Brick>,
    pub ball: Ball,
    pub panel: Panel,
    pub finished: bool,
    pub score: u32,
}

impl BreakoutMechanics {
    pub fn new() -> Self {
        Self {
            bricks: BreakoutMechanics::initial_bricks(),
            ball: BreakoutMechanics::initial_ball(),
            panel: BreakoutMechanics::initial_panel(),
            finished: false,
            score: 0,
        }
    }

    fn initial_bricks() -> Vec<Brick> {
        fn create_brick(left_x: f32, upper_y: f32) -> Brick {
            Brick {
                shape: AaBB {
                    min: Pos2::new(left_x, upper_y - BRICK_EDGE_LEN),
                    max: Pos2::new(left_x + BRICK_EDGE_LEN, upper_y),
                },
            }
        }

        let mut bricks = vec![];
        for row in 0..BRICKS_SETUP_ROWS {
            let mut left_x = BRICKS_SETUP_DISTANCE_LEFT_WALL;
            let upper_y = BRICKS_SETUP_FIRST_ROW_TOP_Y + row as f32 * (BRICK_EDGE_LEN + BRICKS_SETUP_SPACING);
            loop {
                let brick = create_brick(left_x, upper_y);
                if brick.shape.max.x >= MODEL_GRID_LEN_X - BRICKS_SETUP_MIN_DISTANCE_RIGHT_WALL {
                    break;
                } else {
                    left_x = brick.shape.max.x + BRICKS_SETUP_SPACING;
                    bricks.push(brick);
                }
            }
        }
        bricks
    }

    fn initial_ball() -> Ball {
        Ball {
            shape: Circle {
                center: Pos2::new(MODEL_GRID_LEN_X * 0.5, MODEL_GRID_LEN_Y * 0.5),
                radius: BALL_RADIUS,
            },
            direction: Vec2::from((rand::thread_rng().gen_range(-0.35..-0.15), -1.0)),
            speed_per_sec: BALL_SPEED_PER_SEC,
        }
    }

    fn initial_panel() -> Panel {
        Panel {
            shape: AaBB {
                min: Pos2::new(
                    MODEL_GRID_LEN_X / 2.0 - PANEL_LEN_X / 2.0,
                    PANEL_CENTER_POS_Y - PANEL_LEN_Y / 2.0,
                ),
                max: Pos2::new(
                    MODEL_GRID_LEN_X / 2.0 + PANEL_LEN_X / 2.0,
                    PANEL_CENTER_POS_Y + PANEL_LEN_Y / 2.0,
                ),
            },
            speed_per_sec: 0.0,
        }
    }

    /// physically move one time step forward
    pub fn time_step(&mut self, input: GameInput) {
        self.panel.proceed();
        self.proceed_ball_with(self.ball.move_vector());
        self.check_game_end_situation();
        if !self.finished {
            self.panel.process_input(input);
        }
    }

    fn check_game_end_situation(&mut self) {
        if self.ball.shape.center.y >= self.panel.shape.max.y {
            self.finished = true;
        } else if self.bricks.is_empty() {
            self.finished = true;
        }
    }

    fn proceed_ball_with(&mut self, move_vector: Vec2) {
        if move_vector.length() < SPACE_GRANULARITY {
            return;
        }
        let ball = &self.ball;
        assert!(ball.speed_per_sec > 0.0);

        let collisions = self.check_collisions(move_vector);

        // remove brick(s), which have been really hit
        for brick_idx in collisions.surfaces.iter()
            .filter_map(|e| match e.object {
                Some(BreakoutObject::Brick { idx }) => Some(idx),
                _ => None,
            })
            .sorted_unstable()
            .rev()
        {
            self.bricks.remove(brick_idx);
            self.score += 1;
        }

        let ball = &mut self.ball;
        if let Some(collision) = collisions.effective_collision_surface() {
            let collision_center_pos = ball.shape.center + ball.direction * collision.way;
            let remaining_distance = move_vector.length() - collision.way;
            let reflected_direction: Vec2 = reflected_vector(ball.direction, collision.surface_normal).normalized();
            ball.shape.center = collision_center_pos;
            ball.direction = reflected_direction;
            let remaining_move_vector = reflected_direction * remaining_distance;
            log::debug!("move_vector: {:?}, collision: {:?}, remaining_move_vector: {:?}", &move_vector,collision, &remaining_move_vector);
            if remaining_move_vector.length() > 0.0 {
                self.proceed_ball_with(remaining_move_vector);
            }
        } else {
            ball.shape.center = ball.shape.center + move_vector
        }
    }

    fn check_collisions(&self, move_vector: Vec2) -> ContactCandidates {
        // test collisions and keep the one(s) with shortest distance
        let mut collision_candidates = ContactCandidates::new();

        if let Some(c) = self.ball.collision_test_left_wall(move_vector) {
            collision_candidates.consider(ContactObjectSurface::of(c, None));
        }
        if let Some(c) = self.ball.collision_test_right_wall(move_vector) {
            collision_candidates.consider(ContactObjectSurface::of(c, None));
        }
        if let Some(c) = self.ball.collision_test_top_wall(move_vector) {
            collision_candidates.consider(ContactObjectSurface::of(c, None))
        }

        if let Some(c) = self.ball.collision_check_with_rectangle(move_vector, &self.panel.shape) {
            collision_candidates.consider(ContactObjectSurface::of(c, None));
        }

        for (idx, brick) in self.bricks.iter().enumerate() {
            if let Some(c) = self.ball.collision_check_with_rectangle(move_vector, &brick.shape) {
                collision_candidates.consider(ContactObjectSurface::of(c, Some(BreakoutObject::Brick { idx })));
            }
        }
        collision_candidates
    }
}

#[derive(Copy, Clone)]
pub struct GameInput {
    pub control: PanelControl,
    pub exit: bool,
}

impl GameInput {
    pub fn none() -> Self {
        Self {
            control: PanelControl::None,
            exit: false,
        }
    }

    pub fn action(control: PanelControl) -> Self {
        Self {
            control,
            exit: false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PanelControl {
    None,
    AccelerateLeft,
    AccelerateRight,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Brick {
    pub shape: AaBB,
}

/// A ball is a perfect round 2D structure
#[derive(Clone, Debug)]
pub struct Ball {
    pub shape: Circle,
    pub direction: Vec2,
    pub speed_per_sec: f32,
}

impl Ball {
    fn move_vector(&self) -> Vec2 {
        self.direction.normalized() * self.speed_per_sec * TIME_GRANULARITY.as_secs_f32()
    }

    fn collision_test_left_wall(&self, move_vector: Vec2) -> Option<ContactSurface> {
        let wall_distance_x = self.shape.center.x - self.shape.radius;
        assert!(wall_distance_x >= 0.0);

        if wall_distance_x + move_vector.x > 0.0 {
            None
        } else {
            let way_to_collision = move_vector * (wall_distance_x / move_vector.x.abs());
            Some(ContactSurface {
                way: way_to_collision.length(),
                approximation: 0.0,
                surface_normal: Vec2::new(1.0, 0.0),
            })
        }
    }

    fn collision_test_right_wall(&self, move_vector: Vec2) -> Option<ContactSurface> {
        let wall_distance_x = MODEL_GRID_LEN_X - self.shape.center.x - self.shape.radius;
        assert!(wall_distance_x >= 0.0);

        if move_vector.x < wall_distance_x {
            None
        } else {
            let way_to_collision = move_vector * (wall_distance_x / move_vector.x.abs());
            Some(ContactSurface {
                way: way_to_collision.length(),
                approximation: 0.0,
                surface_normal: Vec2::new(-1.0, 0.0),
            })
        }
    }

    fn collision_test_top_wall(&self, move_vector: Vec2) -> Option<ContactSurface> {
        let wall_distance_y = self.shape.center.y - self.shape.radius - CEILING_HEIGHT_Y;
        assert!(wall_distance_y >= 0.0);

        if wall_distance_y + move_vector.y > 0.0 {
            None
        } else {
            let way_to_collision = move_vector * (wall_distance_y / move_vector.y.abs());
            Some(ContactSurface {
                way: way_to_collision.length(),
                approximation: 0.0,
                surface_normal: Vec2::new(0.0, 1.0),
            })
        }
    }

    /// https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
    fn collision_check_with_rectangle(
        &self,
        move_vector: Vec2,
        aabb: &AaBB,
    ) -> Option<ContactSurface> {
        match self.find_non_penetrating_collision(move_vector, aabb) {
            None => None,
            c @ Some(collision) => {
                // only accept collisions, which are +- 90° from move_vector
                if vector_angle(move_vector, collision.surface_normal).abs() > FRAC_PI_2 {
                    c
                } else {
                    // still the previous collision in range; we see the ball was already reflected
                    None
                }
            }
        }
    }

    fn find_non_penetrating_collision(&self, move_vector: Vec2, aabb: &AaBB) -> Option<ContactSurface> {
        // calculate relevant move_distance to go back, based on penetration depth
        // p = penetration depth
        // n1 = circle surface normale (normalized)
        // mv = move_vector
        // alpha = winkel zwischen n1 und mv
        // Formulas: cos(alpha) = p / x
        //           cos(alpha) = n1 ⋅ mv / (n1.len()=1) * mv.len()
        // p / x = n1 ⋅ mv / mv.len()
        // x = p / (n1 ⋅ mv / mv.len())
        fn moved_distance_after_collision(p: f32, n1: Vec2, mv: Vec2) -> f32 {
            debug_assert!(n1.length() > 0.99 && n1.length() < 1.01);
            p / (n1.dot(mv) / mv.length())
        }

        // we know there is a penetrating contact at search_range.end, so there must be non-penetrating contact before
        fn binary_search_first_contact(ball: &Circle, move_vector: Vec2, search_range: Range<f32>, aabb: &AaBB, depth: usize) -> ContactSurface {
            if depth > 10 {
                log::warn!("binary_search_first_contact depth={depth}")
            }
            let m = (search_range.start + search_range.end) / 2.0;
            match contact_test_circle_aabb(
                &Circle {
                    center: ball.center + move_vector * m,
                    radius: ball.radius,
                },
                aabb,
            ) {
                None => binary_search_first_contact(ball, move_vector, m..search_range.end, aabb, depth + 1),
                Some(contact) if contact.dist < -CONTACT_PENETRATION_LIMIT => binary_search_first_contact(ball, move_vector, search_range.start..m, aabb, depth + 1),
                Some(contact) => ContactSurface {
                    way: move_vector.length() * m,
                    approximation: contact.dist,
                    surface_normal: Vec2::new(contact.normal2.x, contact.normal2.y),
                }
            }
        }

        match contact_test_circle_aabb(
            &Circle {
                center: self.shape.center + move_vector,
                radius: self.shape.radius,
            },
            aabb,
        ) {
            None => None,
            Some(contact) if contact.dist < -CONTACT_PENETRATION_LIMIT => {
                let x = moved_distance_after_collision(
                    contact.dist.abs(),
                    Vec2::new(contact.normal1.x, contact.normal1.y),
                    move_vector,
                );
                let estimated_move_vector_portion_till_contact = 1.0 - x / move_vector.length();

                match contact_test_circle_aabb(
                    &Circle {
                        center: self.shape.center + move_vector * estimated_move_vector_portion_till_contact,
                        radius: self.shape.radius,
                    },
                    aabb,
                ) {
                    None => {
                        log::debug!("estimated contact not there - need to performing binary search");
                        Some(binary_search_first_contact(&self.shape, move_vector, estimated_move_vector_portion_till_contact..1.0, aabb, 0))
                    }
                    Some(contact) if contact.dist < -CONTACT_PENETRATION_LIMIT => {
                        log::debug!("estimated contact is still penetrating: dist={}", contact.dist);
                        Some(binary_search_first_contact(&self.shape, move_vector, 0.0..estimated_move_vector_portion_till_contact, aabb, 0))
                    }
                    Some(contact) => Some(ContactSurface {
                        way: move_vector.length() * estimated_move_vector_portion_till_contact,
                        approximation: contact.dist,
                        surface_normal: Vec2::new(contact.normal2.x, contact.normal2.y),
                    }),
                }
            }
            Some(contact) => Some(ContactSurface {
                way: move_vector.length(),
                approximation: contact.dist,
                surface_normal: Vec2::new(contact.normal2.x, contact.normal2.y),
            }),
        }
    }
}

pub struct ContactCandidates {
    surfaces: Vec<ContactObjectSurface>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContactObjectSurface {
    pub way: f32,
    /// final distance after the way
    pub approximation: f32,
    // ⊥ surface normal vector; perpendicular to surface ; normalized
    pub surface_normal: Vec2,
    pub object: Option<BreakoutObject>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BreakoutObject {
    Brick { idx: usize }
}

impl ContactObjectSurface {
    pub fn of(surface: ContactSurface, object: Option<BreakoutObject>) -> Self {
        ContactObjectSurface {
            way: surface.way,
            approximation: surface.approximation,
            surface_normal: surface.surface_normal,
            object,
        }
    }
}

impl From<&ContactObjectSurface> for ContactSurface {
    fn from(value: &ContactObjectSurface) -> Self {
        ContactSurface {
            way: value.way,
            approximation: value.approximation,
            surface_normal: value.surface_normal,
        }
    }
}

impl ContactCandidates {
    pub fn new() -> Self {
        Self {
            surfaces: Vec::with_capacity(2),
        }
    }

    pub fn consider(&mut self, candidate: ContactObjectSurface) {
        fn keep_shortest_way_plus_approximation(surfaces: &mut Vec<ContactObjectSurface>) {
            fn path_len(e: &ContactObjectSurface) -> f32 {
                e.way + e.approximation
            }

            let shortest_path = surfaces.iter()
                .map(|e| path_len(e))
                .fold(f32::INFINITY, |r, len| match len < r {
                    false => r,
                    true => len,
                });

            surfaces.retain(|e| path_len(e) <= shortest_path + SPACE_GRANULARITY);
        }

        assert!(candidate.approximation >= -CONTACT_PENETRATION_LIMIT && candidate.approximation <= CONTACT_PREDICTION);
        self.surfaces.push(candidate);
        if self.surfaces.len() > 1 {
            keep_shortest_way_plus_approximation(&mut self.surfaces)
        }
    }

    /// We may have hit multiple surfaces, so here we determine the effective collision surface for a reflection
    pub fn effective_collision_surface(&self) -> Option<ContactSurface> {
        match self.surfaces.len() {
            0 => None,
            1 => Some(self.surfaces.first().unwrap().into()),
            _ => {
                let normale = self.surfaces.iter()
                    .fold(Vec2::new(0.0, 0.0), |sum, e| sum + e.surface_normal)
                    .normalized();
                let distance = self.surfaces.iter()
                    .fold(0.0, |sum, e| sum + e.approximation)
                    / self.surfaces.len() as f32;
                let way = self.surfaces.iter()
                    .fold(0.0, |sum, e| sum + e.way)
                    / self.surfaces.len() as f32;
                Some(ContactSurface {
                    way,
                    approximation: distance,
                    surface_normal: normale,
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Panel {
    pub shape: AaBB,
    pub speed_per_sec: f32,
}

impl Panel {
    /// calculate new panel move-vector based on input or slow-down
    pub fn process_input(&mut self, input: GameInput) {
        match input.control {
            PanelControl::None => {
                self.speed_per_sec =
                    decrease_speed(self.speed_per_sec, PANEL_SLOW_DOWN_ACCEL_PER_SECOND)
            }
            PanelControl::AccelerateLeft => {
                self.speed_per_sec = accelerate(
                    self.speed_per_sec,
                    -PANEL_CONTROL_ACCEL_PER_SECOND,
                    PANEL_MAX_SPEED_PER_SECOND,
                )
            }
            PanelControl::AccelerateRight => {
                self.speed_per_sec = accelerate(
                    self.speed_per_sec,
                    PANEL_CONTROL_ACCEL_PER_SECOND,
                    PANEL_MAX_SPEED_PER_SECOND,
                )
            }
        }
    }
}

impl Panel {
    /// physically move one time step forward; project new position according to movement
    pub fn proceed(&mut self) {
        // let most_left_center_pos_x = (self.shape.size.x / 2.0).round();
        // let most_right_center_pos_x = (MODEL_GRID_LEN_X - (self.size.x / 2.0)).round();
        let potential_pos = self.shape.translate(Vec2::new(
            self.speed_per_sec * TIME_GRANULARITY.as_secs_f32(),
            0.0,
        ));

        if potential_pos.min.x <= 0.0 {
            self.shape = potential_pos.translate(Vec2::new(-potential_pos.min.x, 0.0));
            self.speed_per_sec = 0.0;
        } else if potential_pos.max.x >= MODEL_GRID_LEN_X {
            self.shape =
                potential_pos.translate(Vec2::new(MODEL_GRID_LEN_X - potential_pos.max.x, 0.0));
            self.speed_per_sec = 0.0;
        } else {
            self.shape = potential_pos;
        }
    }
}

pub trait Assert {
    fn assert(&self);
}

impl Assert for Panel {
    fn assert(&self) {
        assert!(self.shape.min.x >= 0.0);
        assert!(self.shape.max.x <= MODEL_GRID_LEN_X);
        assert!(self.shape.min.y >= 0.0);
        assert!(self.shape.max.y <= MODEL_GRID_LEN_Y);
    }
}

impl Assert for Ball {
    fn assert(&self) {
        assert!(self.shape.center.x - self.shape.radius >= 0.0);
        assert!(self.shape.center.x + self.shape.radius <= MODEL_GRID_LEN_X);
        assert!(self.shape.center.y - self.shape.radius >= 0.0);
        assert!(self.shape.center.y + self.shape.radius <= MODEL_GRID_LEN_Y);
    }
}

fn granulate_speed(speed: f32) -> f32 {
    (speed * 1000.0).round() / 1000.0
}

/// speed: LEN per TP
/// break_acceleration: LEN per TP²  (a positive amount)
fn decrease_speed(speed_per_sec: f32, break_acceleration_per_sec: f32) -> f32 {
    assert!(break_acceleration_per_sec >= 0.0);
    if speed_per_sec > 0.0 {
        (granulate_speed(speed_per_sec - break_acceleration_per_sec)).max(0.0)
    } else if speed_per_sec < 0.0 {
        (granulate_speed(speed_per_sec + break_acceleration_per_sec)).max(0.0)
    } else {
        0.0
    }
}

/// positive or negative speed and acceleration
fn accelerate(speed_per_sec: f32, acceleration_per_sec: f32, speed_limit_abs: f32) -> f32 {
    assert!(!speed_limit_abs.is_sign_negative());

    let virtual_speed = speed_per_sec + acceleration_per_sec;
    let result_speed = if virtual_speed.abs() > speed_limit_abs {
        match virtual_speed.is_sign_positive() {
            true => speed_limit_abs,
            false => -speed_limit_abs,
        }
    } else {
        virtual_speed
    };

    granulate_speed(result_speed)
}

#[cfg(test)]
mod tests {
    use egui::{Pos2, Vec2};
    use rstest::rstest;

    use super::super::algebra_2d::*;
    use super::super::mechanics::*;

    #[rstest]
    #[case(Pos2::new(10.0, 10.0), 5.0, Vec2::new(- 2.0, 2.0), None)]
    #[case(Pos2::new(5.0, 10.0), 5.0, Vec2::new(- 5.0, 0.0), Some(ContactSurface{ way: 0.0, approximation: 0.0, surface_normal: Vec2::new(1.0, 0.0)}))]
    #[case(Pos2::new(7.0, 7.0), 5.0, Vec2::new(- 5.0, 0.0), Some(ContactSurface{ way: 2.0, approximation: 0.0, surface_normal: Vec2::new(1.0, 0.0)}))]
    fn ball_collision_test_left_wall(
        #[case] center: Pos2,
        #[case] radius: f32,
        #[case] move_vector: Vec2,
        #[case] expected_result: Option<ContactSurface>,
    ) {
        let ball = Ball {
            shape: Circle { center, radius },
            direction: Vec2::new(0.0, 0.0),
            speed_per_sec: 0.0,
        };
        assert_eq!(ball.collision_test_left_wall(move_vector), expected_result);
    }

    #[rstest]
    #[case(Pos2::new(MODEL_GRID_LEN_X - 10.0, 10.0), 5.0, Vec2::new(2.0, 2.0), None)]
    #[case(Pos2::new(MODEL_GRID_LEN_X - 5.0, 10.0), 5.0, Vec2::new(5.0, 0.0), Some(ContactSurface{ way: 0.0, approximation: 0.0, surface_normal: Vec2::new(- 1.0, 0.0)}))]
    #[case(Pos2::new(MODEL_GRID_LEN_X - 7.0, 7.0), 5.0, Vec2::new(5.0, 0.0), Some(ContactSurface{ way: 2.0, approximation: 0.0, surface_normal: Vec2::new(- 1.0, 0.0)}))]
    fn ball_collision_test_right_wall(
        #[case] center: Pos2,
        #[case] radius: f32,
        #[case] move_vector: Vec2,
        #[case] expected_result: Option<ContactSurface>,
    ) {
        let ball = Ball {
            shape: Circle { center, radius },
            direction: Vec2::new(0.0, 0.0),
            speed_per_sec: 0.0,
        };
        assert_eq!(ball.collision_test_right_wall(move_vector), expected_result);
    }

    fn assert_eq_roughly(what: &str, a: f32, b: f32, tolerance: f32) {
        assert!(!tolerance.is_sign_negative());
        assert!((a - b).abs() <= tolerance, "'{what}' difference between {a} and {b} more than {tolerance}"
        );
    }

    #[rstest]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(10.0, 0.0), Pos2::new(150.0, 90.0), Pos2::new(170.0, 110.0), None)]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(5.0, 0.0), Pos2::new(110.0, 90.0), Pos2::new(130.0, 110.0), Some(ContactSurface{ way: 5.0, approximation: 0.0, surface_normal: Vec2::new(- 1.0, 0.0)}))]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(3.0, - 3.0), Pos2::new(100.0, 70.0), Pos2::new(120.0, 93.0), Some(ContactSurface{ way: 2.83, approximation: 0.0, surface_normal: Vec2::new(0.0, 1.0)}))]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(- 8.0, - 8.0), Pos2::new(70.0, 80.0), Pos2::new(90.0, 100.0), Some(ContactSurface{ way: 7.07, approximation: 0.0, surface_normal: Vec2::new(1.0, 0.0)}))]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(- 1.46, - 1.46), Pos2::new(80.0, 80.0), Pos2::new(95.0, 95.0), Some(ContactSurface{ way: 2.07, approximation: 0.0, surface_normal: Vec2::new(1.0, 1.0).normalized()}))]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(- 5.0, - 5.0), Pos2::new(80.0, 80.0), Pos2::new(95.0, 95.0), Some(ContactSurface{ way: 2.07, approximation: 0.0, surface_normal: Vec2::new(1.0, 1.0).normalized()}))]
    #[case(Pos2::new(100.0, 100.0), 5.0, Vec2::new(- 4.2, - 4.2), Pos2::new(80.0, 80.0), Pos2::new(90.0, 90.0), None)]
    fn ball_collision_test_rectangle(
        #[case] center: Pos2,
        #[case] radius: f32,
        #[case] move_vector: Vec2,
        #[case] rect_lower_left: Pos2,
        #[case] rect_upper_right: Pos2,
        #[case] expected_result: Option<ContactSurface>,
    ) {
        let ball = Ball {
            shape: Circle { center, radius },
            direction: Default::default(),
            speed_per_sec: 0.0,
        };
        let rect = AaBB {
            min: rect_lower_left,
            max: rect_upper_right,
        };

        let result = ball.collision_check_with_rectangle(move_vector, &rect);

        assert_eq!(result.is_none(), expected_result.is_none());
        if result.is_some() {
            let result = result.unwrap();
            let expected_result = expected_result.unwrap();
            assert_eq_roughly(
                "surface_normal.x",
                result.surface_normal.x,
                expected_result.surface_normal.x,
                0.01,
            );
            assert_eq_roughly(
                "surface_normal.y",
                result.surface_normal.y,
                expected_result.surface_normal.y,
                0.01,
            );
            assert_eq_roughly("way", result.way, expected_result.way, 0.1);
            assert!(result.approximation >= -CONTACT_PENETRATION_LIMIT && result.approximation < CONTACT_PREDICTION);
        }
    }
}
