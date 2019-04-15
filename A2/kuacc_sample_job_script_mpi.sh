#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=cardiac-sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
##SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=30:00
#SBATCH --output=cardiacsim-%j.out
#SBATCH --mail-type=ALL
# #SBATCH --mail-user=nahmad16@ku.edu.tr

################################################################################
################################################################################

## Load openmpi version 3.0.0
echo "Loading openmpi module ..."
module load openmpi/3.0.0

## Load GCC-7.2.1
echo "Loading GCC module ..."
module load gcc/7.3.0

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Serial version ..."
./cardiacsim -n 400 -t 100

# Different MPI+OpenMP configurations
# [1 + 32] [2 + 16] [4 + 8] [8 + 4] [16 + 2] [32 + 1]

echo "1 MPI + 32 OpenMP"
export OMP_NUM_THREADS=32
mpirun -np 1 ./executable_name -o 32

echo "2 MPI + 16 OpenMP"
export OMP_NUM_THREADS=16
mpirun -np 2 ./executable_name -o 16

#....
echo "Finished with execution!"