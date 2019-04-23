#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=d-nonex-cardiac-sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=short
##SBATCH --exclusive
##SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=30:00
#SBATCH --output=./outputs/d-1-cardiacsim-%j.out
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

echo "1 MPI + 2 OpenMP"
export OMP_NUM_THREADS=2
mpirun -np 1 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 2 -t 100 -x 1 -y 1

echo "2 MPI + 2 OpenMP"
export OMP_NUM_THREADS=2
mpirun -np 2 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 2 -t 100 -x 1 -y 2

echo "4 MPI + 2 OpenMP"
export OMP_NUM_THREADS=2
mpirun -np 4 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 2 -t 100 -x 1 -y 4

echo "8 MPI + 2 OpenMP"
export OMP_NUM_THREADS=2
mpirun -np 8 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 2 -t 100 -x 1 -y 8

echo "16 MPI + 2 OpenMP"
export OMP_NUM_THREADS=2
mpirun -np 16 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 2 -t 100 -x 1 -y 16

echo "1 MPI + 4 OpenMP"
export OMP_NUM_THREADS=4
mpirun -np 1 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 4 -t 100 -x 1 -y 1

echo "2 MPI + 4 OpenMP"
export OMP_NUM_THREADS=4
mpirun -np 2 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 4 -t 100 -x 1 -y 2

echo "4 MPI + 4 OpenMP"
export OMP_NUM_THREADS=4
mpirun -np 4 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 4 -t 100 -x 1 -y 4

echo "8 MPI + 4 OpenMP"
export OMP_NUM_THREADS=4
mpirun -np 8 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 4 -t 100 -x 1 -y 8

echo "1 MPI + 8 OpenMP"
export OMP_NUM_THREADS=8
mpirun -np 1 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 8 -t 100 -x 1 -y 1

echo "2 MPI + 8 OpenMP"
export OMP_NUM_THREADS=16
mpirun -np 2 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 8 -t 100 -x 1 -y 2

echo "4 MPI + 8 OpenMP"
export OMP_NUM_THREADS=8
mpirun -np 4 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 8 -t 100 -x 1 -y 4

echo "1 MPI + 16 OpenMP"
export OMP_NUM_THREADS=16
mpirun -np 1 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 16 -t 100 -x 1 -y 1

echo "2 MPI + 16 OpenMP"
export OMP_NUM_THREADS=16
mpirun -np 2 -bind-to socket -map-by socket ./cardiacsim-openmp -n 1024 -o 16 -t 100 -x 1 -y 2

#....
echo "Finished with execution!"