#!/bin/sh

module load python

source /global/homes/y/yifanch/larpix-control/larpix.venv/bin/activate

f_run="/global/cfs/cdirs/dune/www/data/ModuleX/runlist_1-0_Efield.txt"
dir_output="/global/cfs/cdirs/dune/www/data/ModuleX/packetized/Run_1-0_Efield/"
prefix_ouput="selftrigger_packet_"
while IFS= read -r f_input
do
  echo "$f_input"
  IFS="-"
  read -ra name <<<"$f_input"
  f_output=""
  f_output+="$dir_output"
  f_output+="$prefix_ouput"
  f_output+="${name[1]}"
  echo "$f_output"
  python3 /global/homes/y/yifanch/larpix-control/scripts/convert_rawhdf5_to_hdf5.py -i "$f_input" -o "$f_output"
done < "$f_run"
