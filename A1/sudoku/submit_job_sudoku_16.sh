#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=16-sudoku-jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=sudoku-jobs-16.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Job...!"
echo "==============================================================================="
echo "Running compiled binary..."

#serial version
echo "Serial version3..."
./sudoku_solver 16 4x4_hard_3.csv
echo "Serial version2..."
./sudoku_solver 16 4x4_hard_2.csv
echo "Serial version1..."
./sudoku_solver 16 4x4_hard_1.csv

#parallel version
echo "Parallel version with 1 threads"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version with 2 threads"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version with 4 threads"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version with 8 threads"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version scatter with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,scatter
./sudoku_solver_parallel 16 4x4_hard_3.csv 
echo "Parallel version with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel 16 4x4_hard_3.csv 

#parallel version
echo "Parallel version b with 1 threads"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b with 2 threads"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b with 4 threads"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b with 8 threads"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version scatter b with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,scatter
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv

#parallel version
echo "Parallel version b 25 with 1 threads"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 25 with 2 threads"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 25 with 4 threads"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 25 with 8 threads"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 25 with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 25 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25

#parallel version
echo "Parallel version b 30 with 1 threads"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 30 with 2 threads"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 30 with 4 threads"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 30 with 8 threads"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 30 with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version scatter b 30 with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,scatter
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 30 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30

echo "Serial version Early Spot"
./sudoku_solver_c 16 4x4_hard_3.csv 

#parallel version
echo "Parallel version c with 1 threads"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version c with 2 threads"
export OMP_NUM_THREADS=2
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version c with 4 threads"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version c with 8 threads"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version c with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version scatter c with 16 threads"
export OMP_NUM_THREADS=16 
export KMP_AFFINITY=verbose,granularity=fine,scatter
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 
echo "Parallel version c with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_c 16 4x4_hard_3.csv 

