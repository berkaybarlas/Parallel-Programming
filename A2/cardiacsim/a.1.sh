#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=a-1-nonex-cardiac-sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=short
##SBATCH --exclusive
##SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=30:00
#SBATCH --output=./outputs/1-a-cardiacsim-%j.out
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

echo "Serial version ..."
./cardiacsim-serial -n 1024 -t 100

# Different MPI+OpenMP configurations
# [1 + 32] [2 + 16] [4 + 8] [8 + 4] [16 + 2] [32 + 1]

echo "1 MPI"
mpirun -np 1 ./cardiacsim -n 1024 -t 100 -x 1 -y 1

echo "2 MPI 2 1"
mpirun -np 2 ./cardiacsim -n 1024 -t 100 -x 2 -y 1

echo "2 MPI 1 2"
mpirun -np 2 ./cardiacsim -n 1024 -t 100 -x 2 -y 1

echo "4 MPI 2 2"
mpirun -np 4 ./cardiacsim -n 1024 -t 100 -x 2 -y 2

echo "4 MPI 1 4"
mpirun -np 4 ./cardiacsim -n 1024 -t 100 -x 1 -y 4

echo "4 MPI 4 1"
mpirun -np 4 ./cardiacsim -n 1024 -t 100 -x 4 -y 1

echo "8 MPI 1 8"
mpirun -np 8 ./cardiacsim -n 1024 -t 100 -x 1 -y 8

echo "8 MPI 8 1"
mpirun -np 8 ./cardiacsim -n 1024 -t 100 -x 8 -y 1

echo "8 MPI 2 4"
mpirun -np 8 ./cardiacsim -n 1024 -t 100 -x 2 -y 4

echo "16 MPI 4 4"
mpirun -np 16 ./cardiacsim -n 1024 -t 100 -x 4 -y 4

echo "16 MPI 8 2"
mpirun -np 16 ./cardiacsim -n 1024 -t 100 -x 8 -y 2

echo "16 MPI 2 8"
mpirun -np 16 ./cardiacsim -n 1024 -t 100 -x 2 -y 8

echo "16 MPI 1 16"
mpirun -np 16 ./cardiacsim -n 1024 -t 100 -x 1 -y 16

echo "16 MPI 16 1"
mpirun -np 16 ./cardiacsim -n 1024 -t 100 -x 16 -y 1

echo "32 MPI 1 32"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 1 -y 32

echo "32 MPI 2 16"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 2 -y 16

echo "32 MPI 4 8"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 4 -y 8

echo "32 MPI 8 4"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 8 -y 4

echo "32 MPI 16 2"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 16 -y 2

echo "32 MPI 32 1"
mpirun -np 32 ./cardiacsim -n 1024 -t 100 -x 32 -y 1


#....
echo "Finished with execution!"