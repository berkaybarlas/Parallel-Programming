#!/bin/bash
# 
# CompecTA (c) 2017
# 
# You should only work under the /scratch/users/<username> directory.
#
# Julia job submission script
#
# TODO:
#   - Set name of the job below changing "BWA" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 days.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch mumax3_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=cardiac-cuda
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --qos=users
#SBATCH --gres=gpu:1
#SBATCH --time=120:00
#SBATCH --output=cardiacCuda-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bbarlas15@ku.edu.tr

### Mumax3 only GPU works
#SBATCH --gres=gpu:1
#SBATCH constraint=tesla_k20m
#########################

INPUT="example.mx3"

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load CUDA 9.1
echo "CUDA 9.1 loading.."
module load cuda/9.1

## Load Mumax3 3.9.3
echo "Mumax 3.9.3 loading.."
module load mumax3/3.9.3

echo
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo
echo

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
#serial version
echo "Serial version..."

##echo "Running MUMAX3 command..."
##echo "==========================="
##mumax3 $INPUT
##RET=$?
##echo ""
##echo "RTC exited with return code: $RET"
exit $RET
