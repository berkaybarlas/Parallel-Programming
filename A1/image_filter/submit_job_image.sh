#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=image-blurring-jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=image-blurring-jobs.out

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
echo "Serial version..."
./image_blurring coffee.png

#parallel version
echo "Parallel version with 1 threads A"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 2 threads A"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 4 threads A"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 8 threads A"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 16 threads A B"
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 32 threads A"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel coffee.png

echo "Parallel version with 16 threads B scatter"
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,granularity=fine,scatter
./image_blurring_parallel coffee.png

echo "Cilek"

#serial version
echo "Serial version..."
./image_blurring cilek.png

#parallel version
echo "Parallel version with 1 threads A"
export OMP_NUM_THREADS=1 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 2 threads A"
export OMP_NUM_THREADS=2 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 4 threads A"
export OMP_NUM_THREADS=4 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 8 threads A"
export OMP_NUM_THREADS=8 
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 16 threads A B"
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 32 threads A"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=verbose,granularity=fine,compact
./image_blurring_parallel cilek.png

echo "Parallel version with 16 threads B scatter"
export OMP_NUM_THREADS=16
export KMP_AFFINITY=verbose,granularity=fine,scatter
./image_blurring_parallel cilek.png
