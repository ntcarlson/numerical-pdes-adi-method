/**
 * Matrix data type that enforces cache efficient memory access patterns.
 * Only rows of a row major matrix can be accessed and only columns of a
 * column major matrix can be accessed. The two types must be explicitly 
 * converted to the other.
 */
pub mod matrix {
    pub struct RowMajorMatrix {
        data: Vec<f64>,
        workspace: Vec<f64>,
        dim: (usize, usize),
    }

    pub struct ColMajorMatrix {
        data: Vec<f64>,
        workspace: Vec<f64>,
        dim: (usize, usize),
    }

    impl RowMajorMatrix {
        pub fn new(n : usize, m : usize) -> RowMajorMatrix {
            RowMajorMatrix {
                data: vec![0.0; n*m],
                workspace: vec![0.0; n*m],
                dim: (n, m),
            }
        }

        pub fn iter(&mut self) -> impl Iterator<Item = ((usize, usize), &mut f64)> {
            let (_,m) = self.dim;
            self.data
                .iter_mut()
                .enumerate()
                .map(move |(index, value)| ((index/m, index%m), value))
        }

        pub fn row(&mut self, i : usize) -> &mut [f64] {
            let (_,m) = self.dim;
            &mut self.data[m*i..m*(i+1)]
        }

        pub fn apply_row_op<F>(&mut self, op : F) where
            F : Fn(&mut [f64], &mut [f64]) -> () {
            let (n,m) = self.dim;
            for i in 1..n-1 {
                op(&mut self.data[m*i..m*(i+1)], &mut self.workspace[m*i..m*(i+1)]);
            }
            self.workspace[0..m].clone_from_slice(&self.data[0..m]);
            self.workspace[(n-1)*m..n*m].clone_from_slice(&self.data[(n-1)*m..n*m]);
            std::mem::swap(&mut self.data, &mut self.workspace);
        }

        pub fn add<F>(&mut self, f : F) where
            F : Fn(usize, usize) -> f64 {
            for ((i,j), x) in self.iter() {
                *x += f(i,j);
            }
        }

        pub fn convert(mut self) -> ColMajorMatrix {
            let (n,m) = self.dim;
            transpose(&self.data[..], &mut self.workspace[..], (n,m));
            ColMajorMatrix {
                data: self.workspace,
                workspace: self.data,
                dim: self.dim,
            }
        }

        pub fn print(&self) {
            let (n,m) = self.dim;
            for i in 0..n {
                for j in 0..m {
                    print!("{:>10.6}", self.data[i*m + j]);
                }
                println!("");
            }
        }
    }

    impl ColMajorMatrix {
        pub fn new(n : usize, m : usize) -> ColMajorMatrix {
            ColMajorMatrix {
                data: vec![0.0; n*m],
                workspace: vec![0.0; n*m],
                dim: (n, m),
            }
        }

        pub fn iter(&mut self) -> impl Iterator<Item = ((usize, usize), &mut f64)> {
            let (n,_) = self.dim;
            self.data
                .iter_mut()
                .enumerate()
                .map(move |(index, value)| ((index%n, index/n), value))
        }

        pub fn col(&mut self, j : usize) -> &mut [f64] {
            let (n,_) = self.dim;
            &mut self.data[n*j..n*(j+1)]
        }

        pub fn apply_col_op<F>(&mut self, op : F) where
            F : Fn(&mut [f64], &mut [f64]) -> () {
            let (n,m) = self.dim;
            for j in 1..m-1 {
                op(&mut self.data[n*j..n*(j+1)], &mut self.workspace[n*j..n*(j+1)]);
            }
            self.workspace[0..n].clone_from_slice(&self.data[0..m]);
            self.workspace[n*(m-1)..n*m].clone_from_slice(&self.data[n*(m-1)..n*m]);
            std::mem::swap(&mut self.data, &mut self.workspace);
        }

        pub fn add<F>(&mut self, f : F) where
            F : Fn(usize, usize) -> f64 {
            for ((i,j), x) in self.iter() {
                *x += f(i,j);
            }
        }

        pub fn convert(mut self) -> RowMajorMatrix {
            let (n,m) = self.dim;
            transpose(&self.data[..], &mut self.workspace[..], (m,n));
            RowMajorMatrix {
                data: self.workspace,
                workspace: self.data,
                dim: self.dim,
            }
        }

        pub fn print(&self) {
            let (n,m) = self.dim;
            for i in 0..n {
                for j in 0..m {
                    print!("{:>10.6}", self.data[j*n + i]);
                }
                println!("");
            }
        }
    }

    // TODO: Make this efficient
    fn transpose(A1 : &[f64], A2 : &mut [f64], dim : (usize, usize)) {
        let (n,m) = dim;
        for i in 0..n {
            for j in 0..m {
                A2[j*n + i] = A1[i*m + j];
            }
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn row_to_col() {
            let n = 6;
            let m = 4;
            let mut k = 0.0;
            let mut A = RowMajorMatrix::new(n,m);
            for i in 0..n {
                for j in 0..m {
                    A.data[i*m + j] = k;
                    k += 1.0;
                }
            }

            A.print();

            let mut A = A.convert();
            println!("");
            A.print();


            let mut A = A.convert();
            println!("");
            A.print();

            let test_fn = |x1 : &mut [f64], x2 : &mut [f64]| {
                for i in 0..x1.len() {
                    x2[i] = x1[i];
                }
                x2[0] = 0.0;
            };

            A.apply_row_op(test_fn);
            println!("");
            A.print();

            for ((i,j), &mut x) in A.iter() {
                println!("U({}, {}) = {}", i, j, x);
            }

            let mut A = A.convert();
            println!("");
            for ((i,j), &mut x) in A.iter() {
                println!("U({}, {}) = {}", i, j, x);
            }
        }
    }
}