#!/bin/sh
##
## Job name
#SBATCH --job-name vamsi
#SBATCH -n 100
#SBATCH --mem-per-cpu 8000
##
## Partition name
#SBATCH --partition all-HiPri
##
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

echo "Launching controller"
## check sleep time
$HOME/anaconda2/bin/ipcontroller --ip='*' &
sleep 30
# check sleep time 

echo "Launching engines"
srun $HOME/anaconda2/bin/ipengine &
sleep 30


## Finally your executable line

echo "launching job"
$HOME/anaconda2/bin/python numpy_gen_1.py


echo "done"
