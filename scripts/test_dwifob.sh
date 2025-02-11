# Script for testing the DWIFOB solver: 
use_fast="true"               # If we want to use the faster version of dwifob. 
tolerance="1e-8"              # The error tolerance to be used in the solver:
save_convergence_data="true" # If we want to save convergence results to a .json file. 
save_summary="true"          # If we want to save the summary to a .csv file

# The selected solver: (different versions of dwifob) available options: 
# "dwifob", "+restarts", "+scaling", "+primal_weight", ("+step_size")
solver="+step_size"
restart_scheme="constant"     # Chose between "constant", "PDLP", anything else means no restarts.
restart_frequency=40
dwifob_option="nothing"

# Select the instance: 
INSTANCE="nug08-3rd"
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
# INSTANCE="trivial_lp"
# instance_path=./test/trivial_lp_model.mps

# Select fitting name of experiment:
# experiment_name="${INSTANCE}_test_altA2_${solver}_${tolerance}"
# experiment_name="${INSTANCE}_dwifob_slow_${tolerance}"
experiment_name="${INSTANCE}_dwifob_${solver}_restart=PDLP_${tolerance}"
experiment_name="${INSTANCE}_dwifob_${solver}_restart=${restart_frequency}_${tolerance}"

output_file_base="./results/${experiment_name}"

declare -a max_memory_list=(2 3 4) 
declare -a max_memory_list=(1 2 3 4 5 6 7 10 15 20 30 40) 

#### Below this point there are no more settings: #####
use_steering="true"           # If we want to use DWIFOB, this one should always be true.
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
        --save_convergence_data ${save_convergence_data} \
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
        --save_convergence_data ${save_convergence_data} \
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
        --save_convergence_data ${save_convergence_data} \
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
        --save_convergence_data ${save_convergence_data} \
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
        --save_convergence_data ${save_convergence_data} \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme}
fi

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
if [ "$save_summary" == "true" ]; then 
  echo "Problems solved, storing data in file: ./results/${experiment_name}.csv"
  julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv
fi 

# Removing the temporary files:
rm ./results/layout.json
for max_memory in "${max_memory_list[@]}" 
do
  log_dir_name_suffix="_m=${max_memory}"
  rm -rf ${output_file_base}${log_dir_name_suffix}
done

echo "Done"
