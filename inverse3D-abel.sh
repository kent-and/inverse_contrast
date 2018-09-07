#!/bin/bash
# Job name:
#SBATCH --job-name=kent
#
# Project:
#SBATCH --account=nn9279k
# Wall clock limit:
#SBATCH --time='24:00:00'
#
# Max memory usage per task:
#SBATCH --mem-per-cpu=16000M
#
# Number of tasks (cores):
##SBATCH --nodes=1 --ntasks=4
#SBATCH --ntasks=1
##SBATCH --hint=compute_bound
#SBATCH --cpus-per-task=1
#SBTACH --outfile=ValuesA$1B$2N$3.out
#SBATCH --partition=long
##SBATCH --output=output.$SCRATCH 

## Set up job environment
source /cluster/bin/jobsetup

source ~larsmva/pyadjoint-brain-inversion-mod-fenics-2017.2.abel.intel.conf

# Define what to do when job is finished (or crashes)
cleanup "mkdir -p $HOME/Results"
cleanup "cp -r $SCRATCH/slurm* $PWD" 
echo $PWD
echo "SCRATCH is $SCRATCH"

# Copy necessary files to $SCRATCH
cp mesh_invers_contrast.h5 forward_problem.py main.py U.xdmf $SCRATCH

echo "${@:4}"
cd $SCRATCH
ls
echo $SCRATCH
mpirun --bind-to none python3 main.py --alpha $1 --beta $2 --noise $3 --num $4 --tol "${@:4}"


