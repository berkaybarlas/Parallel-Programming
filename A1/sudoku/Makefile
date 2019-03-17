icc: sudoku_solver.c sudoku_solver_parallel.c
	icc sudoku_solver.c -o sudoku_solver -qopenmp -lm
	icc sudoku_solver_c.c -o sudoku_solver_c -qopenmp -lm
	icc sudoku_solver_parallel.c -o sudoku_solver_parallel -qopenmp -lm
	icc sudoku_solver_parallel_b.c -o sudoku_solver_parallel_b -qopenmp -lm
	icc sudoku_solver_parallel_c.c -o sudoku_solver_parallel_c -qopenmp -lm

gcc: sudoku_solver.c
	gcc -o sudoku_solver sudoku_solver.c -fopenmp -lm
