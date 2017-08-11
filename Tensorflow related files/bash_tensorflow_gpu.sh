#!/bin/sh
##
## Job name
#SBATCH --job-name munch50_gpu
##SBATCH -n 20
#SBATCH --mem-per-cpu 8000
##
## Partition name
#SBATCH --partition gpuq
##
#SBATCH --gres=gpu:7
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

## Estimated Time Required in the format D-HH:MM:SS removed time 
##SBATCH --time 0-00:30:00
##
## Name of output file (console output from your executable is written here)
## %N= node name, %j=job id
#SBATCH -o slurm-%N-%j.out
##
## Name of error file
#SBATCH -e slurm-%N-%j.err
##
## Email the user of job status
## < JobStatusCode> = NONE, BEGIN, END, FAIL, REQUEUE, ALL etc.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pragnakar@gmail.com


## Finally your executable line

echo "launching job"
python /data/scratch/daencs690/new50/tensorflow_gpu.py


echo "done"