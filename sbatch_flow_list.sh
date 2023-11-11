#!/bin/bash

#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=3GB
#SBATCH --account=dune
#SBATCH --constraint=cpu

source /global/homes/y/yifanch/ModX/flow_list1.sh
