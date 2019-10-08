mod matrix;
use std::fs::File;
use std::io::{Write, Result};
fn main() -> std::io::Result<()> { 
    use std::f64::consts::PI;

    let n = 16;
    let dt = 0.1/(n as f64);

    let bvp = adi_method::BVP {
        ic     : |x, y| (-2.0*y).exp()*(PI*x).sin(),
        rhs    : |x, y, t| (PI*PI - 4.5)*(-0.5*t -2.0*y).exp()*(PI*x).sin(),
        exact  : Some(|x, y, t| (-0.5*t - 2.0*y).exp()*(PI*x).sin()),
        domain : ((0.0, 1.0), (0.0, 1.0)),
        grid   : (n, n),
        dt     : dt,
        bc     : (
            |_y,_t| 0.0,                                // BC for x = 0
            |_y,_t| 0.0,                                // BC for x = 1
            |x,t|   (-0.5*t).exp()*(PI*x).sin(),        // BC for y = 0
            |x,t|   (-0.5*t-2.0).exp()*(PI*x).sin(),    // BC for y = 1
        )
    };

    let (mut U, error) = bvp.solve(2.0);
    let mut f = File::create("foo.txt")?;
    for ((i,j), u) in U.iter() {
        let u_exact = (bvp.exact.unwrap())(bvp.to_x(i), bvp.to_y(j), 2.0);
        writeln!(&mut f, "{} {} {} {}", bvp.to_x(i), bvp.to_y(j), u, u_exact+0.1)?;
    }
    if let Some(err) = error {
        println!("Error = {}", err);
    }
    Ok(())

}


mod adi_method {
    use crate::matrix::matrix::*;

    type BC = fn(f64, f64) -> f64;

    pub struct BVP {
        pub ic: fn(f64, f64) -> f64,
        pub rhs:  fn(f64, f64, f64) -> f64,
        pub exact: Option<fn(f64, f64, f64) -> f64>,
        pub domain: ((f64, f64), (f64, f64)),
        pub grid : (usize, usize),
        pub dt : f64,
        pub bc : (BC, BC, BC, BC),
    }

    impl BVP {
        pub fn solve(&self, stop_time : f64) -> (ColMajorMatrix, Option<f64>) {
            let (n,m) = self.grid;
            let dt = self.dt;

            let mut U = ColMajorMatrix::new(n+1,m+1);

            // Apply initial conditions
            for ((i,j), x) in U.iter() {
                *x = (self.ic)(self.to_x(i), self.to_y(j));
            }

            let mut t = dt;
            while t < stop_time {
                let f = |i, j|  {
                    if i == 0 || i == n || j == 0 ||  j == m {
                        0.0
                    } else {
                        (self.rhs)(self.to_x(i), self.to_y(j), t)*dt/2.0
                    }
                };

                U = self.adi_step(U, f, t);

                t += self.dt;
            }

            // Calculate error if exact solution is provided
            let mut max_err : Option<f64> = None;
            if let Some(exact) = self.exact {
                let exact = |i, j| exact(self.to_x(i), self.to_y(j), t-dt);
                let err = U.iter().fold(0.0,
                    |err, ((i,j), x)| f64::max(err, (exact(i,j)-*x).abs()));
                max_err = Some(err);
            }

            (U, max_err)
        }

        fn dx(&self) -> f64 {
            let ((x1, x2), (_, _)) = self.domain;
            let (n,_) = self.grid;
            (x2-x1)/(n as f64)
        }

        fn dy(&self) -> f64 {
            let ((_, _), (y1, y2)) = self.domain;
            let (_,m) = self.grid;
            (y2-y1)/(m as f64)
        }

        fn mu(&self) -> (f64, f64) {
            (self.dt/self.dx().powf(2.0), self.dt/self.dy().powf(2.0))
        }

        pub fn to_x(&self, i : usize) -> f64 {
            let ((x1, _), (_, _)) = self.domain;
            x1 + (i as f64)*self.dx()
        }

        pub fn to_y(&self, j : usize) -> f64 {
            let ((_, _), (y1, _)) = self.domain;
            y1 + (j as f64)*self.dy()
        }

        fn apply_bc_x(&self, U : &mut RowMajorMatrix, t : f64) {
            let (bc_x1, bc_x2, _, _) = self.bc;
            let (n,_) = self.grid;

            for (j, u) in U.row(0).iter_mut().enumerate() {
                *u = bc_x1(self.to_y(j), t);
            }

            for (j,u) in U.row(n).iter_mut().enumerate() {
                *u = bc_x2(self.to_y(j), t);
            }
        }

        fn apply_bc_y(&self, U : &mut ColMajorMatrix, t : f64) {
            let (_, _, bc_y1, bc_y2) = self.bc;
            let (_,m) = self.grid;

            for (i, u) in U.col(0).iter_mut().enumerate() {
                *u = bc_y1(self.to_x(i), t);
            }

            for (i,u) in U.col(m).iter_mut().enumerate() {
                *u = bc_y2(self.to_x(i), t);
            }
        }

        fn adi_step<F>(&self, mut U : ColMajorMatrix, f : F, t : f64) -> ColMajorMatrix 
            where F : Fn(usize, usize) -> f64 {
            let (mu_x, mu_y) = self.mu();

            // Explicit part of first half step 
            // println!("----------------------------------");
            // println!("Start");
            // U.print();
            U.apply_col_op(|x1, x2| explicit_stencil(mu_y, x1, x2));
            U.add(&f);
            // println!("After first explicit half step");
            // U.print();
            self.apply_bc_y(&mut U, t);
            // println!("After y boundary conditions applied");
            // U.print();

            // Implicit part of first half step
            let mut U = U.convert(); // Convert U to row major form
            U.apply_row_op(|x1, x2| implicit_stencil(mu_x, x1, x2));
            // println!("After first implicit half step");
            // U.print();

            // Explicit part of second half step
            U.apply_row_op(|x1, x2| explicit_stencil(mu_x, x1, x2));
            U.add(&f);
            // println!("After second explicit half step");
            // U.print();
            self.apply_bc_x(&mut U, t);
            // println!("After x boundary conditions applied");
            // U.print();
            
            // Implicit part of second half step
            let mut U = U.convert(); // Convert U to column major form
            U.apply_col_op(|x1, x2| implicit_stencil(mu_y, x1, x2));
            // println!("After second implicit half step");
            // U.print();

            return U;
        }
    }


    fn explicit_stencil(mu : f64, x1 : &mut [f64], x2 : &mut [f64]) {
        let n = x1.len();

        x2[0]   = x1[0];
        x2[n-1] = x1[n-1];

        for i in 1..n-1 {
            x2[i] = 0.5*mu*x1[i-1] + (1.0 - mu)*x1[i] + 0.5*mu*x1[i+1];
        }
    }

    fn implicit_stencil(mu : f64, x1 : &mut [f64], x2 : &mut [f64]) {
        let n = x1.len();

        x1[1]   += 0.5*mu*x1[0];
        x1[n-2] += 0.5*mu*x1[n-1];

        x2[0]   = x1[0];
        x2[n-1] = x1[n-1];

        // Use x2's memory to store the diagonal of A during forward reduction
        x2[1] = 1.0 + mu;

        // Forward reduction step
        for i in 2..n-1 {
            let w = -0.5*mu/x2[i-1];
            x2[i] = (1.0 + mu) + 0.5*mu*w;
            x1[i] -= w*x1[i-1];
        }

        // Backwards reduction step
        x2[n-2] = x1[n-2]/x2[n-2];
        for i in (1..n-2).rev() {
            x2[i] = (x1[i] + 0.5*mu*x2[i+1])/x2[i];
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn explicit_step_test() {
            let n = 17;
            let mut x1 : Vec<f64> = vec!(0.0; n);
            let mut x2 : Vec<f64> = vec!(0.0; n);

            let known = [0.000000000000000,
                         0.056250000000000,
                         0.107031250000000,
                         0.150000000000000,
                         0.185156250000000,
                         0.212500000000000,
                         0.232031250000000,
                         0.243750000000000,
                         0.247656250000000,
                         0.243750000000000,
                         0.232031250000000,
                         0.212500000000000,
                         0.185156250000000,
                         0.150000000000000,
                         0.107031250000000,
                         0.056250000000000,
                         0.000000000000000];

            for i in 0..n {
                let x = (i as f64)/((n-1) as f64);
                x1[i] = x*(1.0-x);
            }

            explicit_stencil(0.6, &mut x1[..], &mut x2[..]);

            let epsilon = 1e-10;
            for (&x, &known) in x2.iter().zip(known.iter()) {
                println!("{:.9}", x);
                // assert!(x < known + epsilon);
                // assert!(x > known - epsilon);
            }

        }

        #[test]
        fn implicit_step_test() {
            let n = 17;
            let mut x1 : Vec<f64> = vec!(0.0; n);
            let mut x2 : Vec<f64> = vec!(0.0; n);

            let known = [ 0.000000000000000,
                          0.056706094932006,
                          0.107120006304029,
                          0.150017272022818,
                          0.185159611151001,
                          0.212500654115857,
                          0.232031377466900,
                          0.243750025707610,
                          0.247656259640354,
                          0.243750025707610,
                          0.232031377466900,
                          0.212500654115857,
                          0.185159611151001,
                          0.150017272022818,
                          0.107120006304029,
                          0.056706094932006,
                          0.000000000000000];

            for i in 0..n {
                let x = (i as f64)/((n-1) as f64);
                x1[i] = x*(1.0-x);
            }

            implicit_stencil(0.6, &mut x1[..], &mut x2[..]);

            let epsilon = 1e-10;
            for (&x, &known) in x2.iter().zip(known.iter()) {
                assert!(x < known + epsilon);
                assert!(x > known - epsilon);
            }
        }
    }
}