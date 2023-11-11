#!/usr/bin/env bash

module load python

source /global/homes/y/yifanch/ndlar_flow/flow.venv/bin/activate

if [[ "$NERSC_HOST" == "cori" ]]; then
    export HDF5_USE_FILE_LOCKING=FALSE
fi

cd ../ndlar_flow

# charge workflows
workflow1='yamls/moduleX_flow/workflows/charge/charge_event_building.yaml'
workflow2='yamls/moduleX_flow/workflows/charge/charge_event_reconstruction.yaml'
workflow3='yamls/moduleX_flow/workflows/combined/combined_reconstruction.yaml'
workflow4='yamls/moduleX_flow/workflows/charge/prompt_calibration.yaml'
workflow5='yamls/moduleX_flow/workflows/charge/final_calibration.yaml'

# read files
runlist="/global/cfs/cdirs/dune/www/data/ModuleX/runlist_packet_0-5_Efield_1.txt"
dir_output="/global/cfs/cdirs/dune/www/data/ModuleX/flow/Run_0-5_Efield_1/"
prefix="flow_"

while IFS= read -r f_input
do
  #echo "$f_input"
  IFS="/"
  read -ra name <<<"$f_input"
  f_output=""
  f_output+="$dir_output"
  f_output+="$prefix"
  f_output+="${name[10]}"
  
  rm -f "$f_output"
  
  # Ensure that the second h5flow doesn't run if the first one crashes. This also
  # ensures that we properly report the failure to the production system.
  #set -o errexit

  echo "$f_input"
  echo "$f_output"
  
  #h5flow -c $workflow1 $workflow2 $workflow3 $workflow4 $workflow5\
  h5flow -c "$workflow1" "$workflow2" "$workflow3" "$workflow4"\
      -i "$f_input" -o "$f_output"
done < "$runlist"

