#![allow(unused)]

const BOARD_DIM_X: usize = 30;
const BOARD_DIM_Y: usize = 30;

const PANEL_POS_Y: usize = 29;
const PANEL_SIZE_IN_BLOCKS: usize = 3;
const PANEL_MAX_SPEED: f64 = 3.0;

const PANEL_MAX_SPEED_INCREASE_PER_SEC: f64 = 1.5;
const PIXEL_PER_BLOCK_SCALE: usize = 3;
const SCREEN_DIM_X: usize = BOARD_DIM_X * PIXEL_PER_BLOCK_SCALE;
const SCREEN_DIM_Y: usize = BOARD_DIM_Y * PIXEL_PER_BLOCK_SCALE;


pub struct Vector2d(isize, isize);

pub struct Vector2df(f64, f64);


pub struct PongGame {
    // x = 0 = left side; y = 0 = bottom
    bricks: [[bool; BOARD_DIM_X]; BOARD_DIM_Y],
    ball: Ball,
    panel: Panel,
}

/// A ball is a perfect round 2D structure with a radius of 0.5 blocks
pub struct Ball {
    radius: f64,
    pos: Vector2df,
    direction: Vector2df,
    speed: f64, // block / second
}

pub struct Panel {
    center_pos_x: usize,
    // in block / second
    speed: f64,
}

impl PongGame {
    pub fn new() -> Self {
        PongGame {
            bricks: Self::init_bricks(),
            ball: Self::init_ball(),
            panel: Self::init_panel(),
        }
    }

    pub fn draw_screen(&self) -> [[bool; SCREEN_DIM_X]; SCREEN_DIM_Y] {
        let mut screen = [[false; SCREEN_DIM_X]; SCREEN_DIM_Y];
        self.draw_bricks(&mut screen);
        self.draw_ball(&mut screen);
        self.draw_panel(&mut screen);
        screen
    }

    fn init_bricks() -> [[bool; BOARD_DIM_X]; BOARD_DIM_Y] {
        let mut bricks = [[false; BOARD_DIM_X]; BOARD_DIM_Y];
        for y in BOARD_DIM_Y - 5..BOARD_DIM_Y - 2 {
            for x in 2..BOARD_DIM_X - 3 {
                bricks[y][x] = true;
            }
        }
        bricks
    }

    fn init_ball() -> Ball {
        Ball {
            radius: 0.5,
            pos: Vector2df(BOARD_DIM_X as f64 / 2.0, BOARD_DIM_Y as f64 / 2.0),
            direction: Vector2df(-0.6, -1.0),
            speed: 0.75,
        }
    }

    fn init_panel() -> Panel {
        Panel {
            center_pos_x: (BOARD_DIM_X as f64 / 2.0).round() as usize,
            speed: 0.0,
        }
    }

    fn draw_bricks(&self, screen: &mut [[bool; SCREEN_DIM_X]; SCREEN_DIM_Y]) {
        for y in 0..BOARD_DIM_Y {
            for x in 0..BOARD_DIM_X {
                if self.bricks[y][x] {
                    for off_y in 0..PIXEL_PER_BLOCK_SCALE {
                        for off_x in 0..PIXEL_PER_BLOCK_SCALE {
                            screen[y * PIXEL_PER_BLOCK_SCALE + off_y][x * PIXEL_PER_BLOCK_SCALE + off_x] = true;
                        }
                    }
                }
            }
        }
    }

    fn draw_ball(&self, screen: &mut [[bool; SCREEN_DIM_X]; SCREEN_DIM_Y]) {
        fn inside_circle(x: isize, y: isize, radius: f64) -> bool {
            ((x.pow(2) + y.pow(2)) as f64).sqrt() <= radius
        }

        let middle_pos = Vector2df(self.ball.pos.0 * PIXEL_PER_BLOCK_SCALE as f64, self.ball.pos.1 * PIXEL_PER_BLOCK_SCALE as f64);
        let radius = self.ball.radius;

        let middle_pixel_pos = Vector2d(middle_pos.0.round() as isize, middle_pos.1.round() as isize);
        let block_range = (radius * PIXEL_PER_BLOCK_SCALE as f64).round() as isize;
        for y in -block_range as isize..block_range {
            for x in -block_range..block_range {
                if inside_circle(x, y, radius) {
                    let screen_x = middle_pixel_pos.0 + x;
                    let screen_y = middle_pixel_pos.1 + y;
                    screen[screen_y as usize][screen_x as usize] = true;
                }
            }
        }
    }

    fn draw_panel(&self, screen: &mut [[bool; SCREEN_DIM_X]; SCREEN_DIM_Y]) {
        let pixel_half_range_x = (PANEL_SIZE_IN_BLOCKS as f64 * 0.5 * PIXEL_PER_BLOCK_SCALE as f64).round() as isize;
        for y in 0..PIXEL_PER_BLOCK_SCALE {
            for x in -pixel_half_range_x..pixel_half_range_x {
                let pixel_pos_y = PANEL_POS_Y * PANEL_SIZE_IN_BLOCKS + y;
                assert!((self.panel.center_pos_x * PIXEL_PER_BLOCK_SCALE) as isize + x > 0);
                let pixel_pos_x = ((self.panel.center_pos_x * PIXEL_PER_BLOCK_SCALE) as isize + x) as usize;
                screen[pixel_pos_y][pixel_pos_x] = true;
            }
        }
    }
}
