set terminal pdf font "Times New Roman, 24"
unset key
set view 60, 140, 1, 1.5
set ztics 0,0.1,0.5
set zrange [0:0.4]

set output "solution16.pdf"
set title "Numerical solution at t=2 with N=16"
set dgrid3d 17,17
splot 'solution16.txt' u 1:2:3 with lines

set output "solution32.pdf"
set title "Numerical solution at t=2 with N=32"
set dgrid3d 33,33
splot 'solution32.txt' u 1:2:3 with lines

set output "solution64.pdf"
set title "Numerical solution at t=2 with N=64"
set dgrid3d 65,65
splot 'solution64.txt' u 1:2:3 with lines

set output "solution128.pdf"
set title "Numerical solution at t=2 with N=128"
set dgrid3d 129,129
splot 'solution128.txt' u 1:2:3 with lines
