mod matrix;
mod adi_method;

use std::fs::File;
use std::io::{Write, Result};
fn main() -> std::io::Result<()> { 
    use std::f64::consts::PI;

    let stop_time = 2.0;
    let mut errors : Vec<f64> = Vec::new();
    for i in 4..=7 {
        let n = 2usize.pow(i);
        let dt = 0.1/(n as f64);

        let bvp = crate::adi_method::adi_method::BVP {
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


        let (mut U, error) = bvp.solve(stop_time);
        let mut f = File::create(format!("analysis/solution{}.txt", n))?;
        for ((i,j), u) in U.iter() {
            writeln!(&mut f, "{:>10.8} {:>10.8} {:>10.8}", bvp.to_x(i), bvp.to_y(j), u)?;
        }
        if let Some(err) = error {
            errors.push(err);
            println!("Error = {}", err);
        }
    }

    for (k, error) in errors.iter().enumerate() {
        print!("e_{} = {},", k, error);
        if let Some(next_error) = errors.get(k+1) {
            let ratio = errors[k]/errors[k+1];
            println!("e_{}/e_{} = {}", k, k+1, ratio, )
        } else {
            println!("");
        }
    }
    Ok(())
}
