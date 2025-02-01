# Script for testing the DWIFOB solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"

# The selected solver: (either pdhg or dwifob)
solver="dwifob"

# instance_path=./test/trivial_lp_model.mps
# experiment_name="trivial_test_${solver}_${tolerance}"

INSTANCE="nug08-3rd"
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
experiment_name="${INSTANCE}_test_power_step_${solver}_${tolerance}"

output_dir="./results/trivial_LP"
output_file_base="${output_dir}/${solver}_solve_trivial_${tolerance}"


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
        --steering_vectors ${use_steering} \
        --iteration_limit 5000

echo "Problems solved, storing data in csv format..."

# Creating the JSON for collecting the results using another Julia Script:
json_content='{"datasets": ['

# declare -a step_size_list=(0.99 0.5 0.1 0.05 0.01 0.005) 
# declare -a max_memory_list=(1 2 3 5 10)

# NOTE: When changing these lists: 
# NOTE: do not forget to change the ones in test_solver.jl as well.
declare -a max_memory_list=(1 2 3 5 10 15)
declare -a step_size_list=("power_iteration") 

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
