# Script for testing the DWIFOB solver: 
use_fast="true"
use_steering="true"
# This is the error tolerance to be used in the solver:
tolerance="1e-4"
# The selected solver: (different versions of dwifob) available options: 
# "dwifob", "+restarts", "+scaling" (In the future: "+primal_weight", "+step_size")
solver="+primal_weight"
restart_scheme="constant"
restart_frequency=40
dwifob_option="alt_A"
# Select the instance: 
INSTANCE="nug08-3rd"
# Select fitting name of experiment:
# experiment_name="${INSTANCE}_test_altA2_${solver}_${tolerance}"
experiment_name="${INSTANCE}_test_${solver}_restart=${restart_frequency}_${tolerance}"

#### Below this point there are no more settings: #####
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
output_file_base="./results/${experiment_name}"

declare -a max_memory_list=(1 2 3 4 5 10 20 40) 
# declare -a max_memory_list=(1 2 3 4 5 10 15 20 30 40 50 60) 

# Bulding a string for the max memory input to the julia program: 
max_memory_input="["
for max_memory in "${max_memory_list[@]}" 
do
  max_memory_input="${max_memory_input}${max_memory}, "
done
max_memory_input="${max_memory_input::-2}]"

echo "Solving ${INSTANCE} with ${solver}..."

if [ "$solver" == "dwifob" ]; then # This is the baseline vanilla dwifob:   
  julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy "constant" \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --restart_scheme "no_restart" \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_option ${dwifob_option}

elif [ "$solver" == "+restarts" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency}

elif [ "$solver" == "+scaling" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency}
        
elif [ "$solver" == "+primal_weight" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme}

elif [ "$solver" == "+step_size" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme}
fi

echo "Problems solved, storing data in csv format..."
# Creating the JSON for collecting the results using another Julia Script:
json_content='{"datasets": ['
for max_memory in "${max_memory_list[@]}" 
do
  log_dir_name_suffix="_m=${max_memory}"
  json_content=${json_content}$'
    {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'", "memory": "'${max_memory}'"},
      "logs_directory": "'${output_file_base}${log_dir_name_suffix}'"},'
done

json_content=${json_content::-1}' 
], "config_labels": ["solver", "tolerance", "memory"]}'

echo "$json_content" > ./results/layout.json

## Storing the results in CSV file using the process_json_to_csv.jl script:
julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv

# Removing the temporary files:
rm ./results/layout.json
for max_memory in "${max_memory_list[@]}" 
do
  log_dir_name_suffix="_m=${max_memory}"
  rm -rf ${output_file_base}${log_dir_name_suffix}
done

echo "Done"
