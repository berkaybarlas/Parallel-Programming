#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=32-4-sudoku
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=sudoku-jobs-32-4.out

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
echo "Parallel version b-hard3 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv
echo "Parallel version b-hard1 20 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv
echo "Parallel version b-hard2 20 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv

#parallel version

echo "Parallel version b 20 with 32 threads hard3"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 20
echo "Parallel version b 25 with 32 threads hard3"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 25
echo "Parallel version b 30 with 32 threads hard3"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b 40 with 32 threads hard3"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 40

echo "Parallel version b 20 with 32 threads hard2"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv 20
echo "Parallel version b 25 with 32 threads hard2"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv 25
echo "Parallel version b 30 with 32 threads hard2"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv 30
echo "Parallel version b 40 with 32 threads hard2"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv 40

echo "Parallel version b 20 with 32 threads hard1"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv 20
echo "Parallel version b 25 with 32 threads hard1"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv 25
echo "Parallel version b 30 with 32 threads hard1"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv 30
echo "Parallel version b 40 with 32 threads hard1"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv 40


#parallel version
echo "Parallel version b-hard3 30 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_3.csv 30
echo "Parallel version b-hard1 30 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_1.csv 30
echo "Parallel version b-hard2 30 with 32 threads"
export OMP_NUM_THREADS=32 
export KMP_AFFINITY=verbose,granularity=fine,compact
./sudoku_solver_parallel_b 16 4x4_hard_2.csv 30
