#!/usr/bin/env bash
#
runlist="/global/cfs/cdirs/dune/www/data/ModuleX/runlist/runlist_flow_0-5_Efield_2.txt"
dir_output="/global/cfs/cdirs/dune/www/data/ModuleX/analysis_data/analysis_0-5_Efield_2/"
prefix="hit_seg_edge_"

TEMPLATE=sbatch_flow2npz_single_temp.sh

while IFS= read -r f_input
do
  IFS="/"
  read -ra name <<<"$f_input"
  f_output=""
  f_output+="$dir_output"
  f_output+="$prefix"
  flow_name="${name[10]}"
  f_output+="${flow_name:24:21}" # time label
  f_output+=".npz"

  if [ -f "$f_output" ]
  then
    echo "$f_output exists."
    continue
  fi

  f_scratch_input="$SCRATCH"
  f_scratch_input+="/ModX/"

  cp "$f_input" "$f_scratch_input"

  f_scratch_input+="$flow_name"

  THIS_TEMP=${TEMPLATE/_temp.sh/_${flow_name:24:21}.sh}
  cp ${TEMPLATE} ${THIS_TEMP}

  sed -i -e "s/INFILE/${f_scratch_input//'/'/'\/'}/g" ${THIS_TEMP}
  sed -i -e "s/OUTFILE/${f_output//'/'/'\/'}/g" ${THIS_TEMP}

  sbatch ${THIS_TEMP}

  echo "input: $f_input"
  echo "output: $f_output"

  rm ${THIS_TEMP}

done < "$runlist"


