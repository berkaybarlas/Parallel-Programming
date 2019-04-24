#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=e-nonex-cardiac-sim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --partition=short
##SBATCH --exclusive
##SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=02:00:00
#SBATCH --output=./outputs/e-cardiacsim-%j.out
#SBATCH --mail-type=ALL
# #SBATCH --mail-user=bbarlas15@ku.edu.tr
#SBATCH --mem-per-cpu=1000M


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
#env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

# Different MPI+OpenMP configurations
# [1 + 32] [2 + 16] [4 + 8] [8 + 4] [16 + 2] [32 + 1]


echo "Serial version ..."
./cardiacsim-serial -n 256 -t 404
echo "1 MPI n=256"
mpirun -np 1 ./cardiacsim -n 256 -t 404 -x 1 -y 4

echo "Serial version ..."
./cardiacsim-serial -n 512 -t 152
echo "2 MPI n=256"
mpirun -np 2 ./cardiacsim -n 512 -t 152 -x 1 -y 2

echo "Serial version ..."
./cardiacsim-serial -n 1024 -t 44
echo "4 MPI n=256"
mpirun -np 4 ./cardiacsim -n 1024 -t 44 -x 1 -y 4

echo "Serial version ..."
./cardiacsim-serial -n 2048 -t 12
echo "8 MPI n=256"
mpirun -np 8 ./cardiacsim -n 2048 -t 12 -x 1 -y 8

echo "Serial version ..."
./cardiacsim-serial -n 4096 -t 3
echo "16 MPI n=256"
mpirun -np 16 ./cardiacsim -n 4096 -t 3 -x 1 -y 16

echo "Serial version ..."
./cardiacsim-serial -n 8192 -t 1
echo "32 MPI n=256"
mpirun -np 32 ./cardiacsim -n 8192 -t 1 -x 1 -y 32

#....
echo "Finished with execution!"