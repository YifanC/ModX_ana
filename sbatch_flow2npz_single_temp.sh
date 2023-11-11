#!/bin/bash

#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=3GB
#SBATCH --account=dune
#SBATCH --constraint=cpu

module load python

source /global/homes/y/yifanch/ndlar_flow/flow.venv/bin/activate

python3 /global/homes/y/yifanch/ModX/flow2npz_single.py --infile INFILE --outfile OUTFILE

