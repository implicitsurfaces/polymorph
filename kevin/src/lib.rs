use fidget::{
    eval::Shape,
    render::{BitRenderMode, RenderConfig},
};

pub struct Bitmap {
    pub pixels: Vec<bool>, // row major order
    pub width: usize,
    pub height: usize,
}

pub fn render(shape: impl Shape) -> Bitmap {
    let image_size = 256;

    let mut cfg = RenderConfig::<2> {
        image_size,
        ..RenderConfig::default()
    };

    // Need to set bounds so our rendered image always has total area 1.
    cfg.bounds.size = 0.5;

    let mut b = Bitmap {
        pixels: Vec::with_capacity(image_size * image_size),
        width: image_size,
        height: image_size,
    };

    let out = cfg.run(shape, &BitRenderMode).unwrap();

    b.pixels.extend(out.iter());
    b
}

//Assumes a single contiguous shape.

impl Bitmap {
    pub fn volume(&self) -> f64 {
        self.pixels.iter().filter(|x| **x).count() as f64 / (self.width * self.height) as f64
    }
    pub fn perimeter(&self) -> f64 {
        assert!(!self.touches_bounds());
        // calculate perimeter by counting how many cardinal direction neighbors are empty.
        // Don't count pixels on bounds, as these are asserted empty above.

        // TODO: I don't think I can count separately, need corners to be sqrt 2 length rather than 2
        let mut total = 0;
        for j in 1..self.height - 2 {
            for i in 1..self.width - 2 {
                if self.pixel(i, j) {
                    total += self.pixel(i - 1, j) as usize; // N
                    total += self.pixel(i + 1, j) as usize; // S
                    total += self.pixel(i, j + 1) as usize; // E
                    total += self.pixel(i, j - 1) as usize; // W
                }
            }
        }
        total as f64 / (self.width * self.height) as f64
    }

    pub fn row(&self, i: usize) -> &[bool] {
        &self.pixels[(i * self.width)..((i + 1) * self.width)]
    }

    pub fn pixel(&self, i: usize, j: usize) -> bool {
        self.pixels[j * self.width + i]
    }

    pub fn touches_bounds(&self) -> bool {
        // anything set in first row
        if self.row(0).iter().any(|x| *x) {
            return true;
        }
        //anything set in last row
        if self.row(self.height - 1).iter().any(|x| *x) {
            return true;
        }

        //anything set in first or last column
        for j in 0..self.height - 1 {
            let r = self.row(j);
            if r[0] || r[self.width - 1] {
                return true;
            }
        }

        return false;
    }

    pub fn print(&self) {
        let mut iter = self.pixels.iter();
        for _y in 0..self.height {
            for _x in 0..self.width {
                if *iter.next().unwrap() {
                    print!("X");
                } else {
                    print!(" ");
                }
            }
            print!("\n");
        }
    }
}
