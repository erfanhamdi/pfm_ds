#!/bin/bash -l

#$ -pe mpi_16_tasks_per_node 16
#$ -N mpi_job
#$ -l h_rt=12:00:00
# Merge stderr into the stdout file, to reduce clutter.
#$ -j y
#$ -m beas
## end of qsub options
# openmpi is the standard MPI library
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load openmpi
module load miniconda
conda activate fenicsx-env
# mpirun -np $NSLOTS python3 2d_shear.py
# mpirun -np $NSLOTS python3 <file name that user provides in command line>
echo "python file: $1"
echo "model: $2"
echo "mesh size: $3"
echo "out file: $4"
mpirun -np $NSLOTS python3 $1 --model $2 --mesh_size $3 --out_file $4 --job_id $JOB_ID

