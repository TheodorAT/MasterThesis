# Script for testing the DWIFOB solver: 

use_fast="$1"
# This is the error tolerance to be used in the solver:
tolerance="1e-4"
# The selected solver: (either pdhg or dwifob)
solver="dwifob"

INSTANCE="trivial_lp_model"
instance_path=./test/${INSTANCE}.mps
experiment_name="trivial_test_fast_${solver}_${tolerance}"

INSTANCE="nug08-3rd"
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
experiment_name="${INSTANCE}_test_fast_${solver}_${tolerance}"

output_dir="./results/${INSTANCE}"
output_file_base="${output_dir}/${solver}_solve_trivial_${tolerance}"

declare -a max_memory_list=(1)
declare -a step_size_list=("power_iteration") 

# Bulding a string for the max memory input to the julia program: 
max_memory_input="["
for max_memory in "${max_memory_list[@]}" 
do
  max_memory_input="${max_memory_input}${max_memory}, "
done
max_memory_input="${max_memory_input::-2}]"

if [ "$solver" == "pdhg" ]; then
  use_steering="false"
elif [ "$solver" == "dwifob" ]; then
  use_steering="true"
else
  echo "Invalid solver used"
  return
fi

echo "Solving lp with ${solver}..."

julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --restart_scheme "no_restart" \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --scale_invariant_initial_primal_weight false \
        --step_size_policy "constant" \
        --iteration_limit 5000 \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast}

echo "Problems solved, storing data in csv format..."
# Creating the JSON for collecting the results using another Julia Script:
json_content='{"datasets": ['

for max_memory in "${max_memory_list[@]}" 
do
  for step_size in "${step_size_list[@]}" 
  do
    log_dir_name_suffix="_gamma=${step_size}_m=${max_memory}"
    json_content=${json_content}$'
      {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'", "gamma": "'${step_size}'", "memory": "'${max_memory}'"},
        "logs_directory": "'${output_file_base}${log_dir_name_suffix}'"},'
  done
done

json_content=${json_content::-1}' 
], "config_labels": ["solver", "tolerance", "gamma", "memory"]}'

echo "$json_content" > ./results/layout.json

## Storing the results in CSV file using the process_json_to_csv.jl script:
julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv

# # Removing the temporary files:
# rm ./results/layout.json
# rm -rf $output_dir

echo "Done"
