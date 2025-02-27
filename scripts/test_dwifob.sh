# Script for testing the DWIFOB solver: 
use_fast="true"              # If we want to use the faster version of dwifob. 
tolerance="1e-4"              # The error tolerance to be used in the solver:
save_convergence_data="true"  # If we want to save convergence results to a .json file. 
save_detailed="true"          # If we want to save the detailed results to a .json file.
save_summary="true"           # If we want to save the summary to a .csv file
iteration_limit="5000"        # The iteration limit for each solver and problem.

# The selected solver: (different versions of dwifob) available options: 
# "dwifob", "+restarts", "+scaling", "+primal_weight", "+step_size"
solver="+primal_weight"
PDLP_restart_scheme="adaptive_normalized"   # Default: "adaptive_normalized", others: "no_restart", 
                                            # "adaptive_localized", "adaptive_distance"
restart_scheme="constant"                   # Chose between "constant", "PDLP", "NOFOB", anything else means no restarts.
restart_frequency=40
dwifob_option="org"           # Chose between "alt_A", "alt_B", "alt_C", anything else means the original.
termination_eval_freq=1           # The frequency of evaluating if we have reached 
                                  # the solution, this also affects the granularity of the saved results.

# Select the instance: 
# (default: nug08-3rd)
# smallest: self, smaller m better m=(1, 2) best
# smaller: chrom1024-7, <-- This one got numerical error for +step_size (since movement=0), no conclusive results on best m.
# Some more reasonable sized ones: 
# - ts-palko
# - savsched1
# - neos3
# - karted
# larger: buildingenergy, 

INSTANCE="savsched1"
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
# INSTANCE="less_trivial_lp"
# instance_path=./test/less_trivial_lp_model.mps

# Select fitting name of experiment:
# experiment_name="${INSTANCE}_dwifob_slow_${tolerance}"
# experiment_name="${INSTANCE}_dwifob_${solver}_restart=PDLP_${tolerance}"
experiment_name="${INSTANCE}_dwifob_${solver}_${tolerance}"
experiment_name="${INSTANCE}_dwifob_zeta=0.5_${dwifob_option}_${solver}_restart=${restart_frequency}_${tolerance}"
experiment_name="${INSTANCE}_dwifob_${dwifob_option}_${solver}_restart=${restart_frequency}_${tolerance}"
output_file_base="./results/${experiment_name}"

declare -a max_memory_list=(1)
declare -a max_memory_list=(1 2 3 4 5 6 7 10 15 20 30 40) 
declare -a max_memory_list=(1 3 5 10 15 20 40) 
# declare -a max_memory_list=(30 40) # These are the ones that we have not yet done for buildingenergy.

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
        --iteration_limit $iteration_limit \
        --termination_evaluation_frequency ${termination_eval_freq} \
        --step_size_policy "constant" \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --restart_scheme "no_restart" \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --save_convergence_data ${save_convergence_data} \
        --save_detailed_convergence_data ${save_detailed} \
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
        --iteration_limit $iteration_limit \
        --termination_evaluation_frequency ${termination_eval_freq} \
        --step_size_policy constant \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --restart_scheme ${PDLP_restart_scheme} \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --save_convergence_data ${save_convergence_data} \
        --save_detailed_convergence_data ${save_detailed} \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency} \
        --dwifob_option ${dwifob_option}

elif [ "$solver" == "+scaling" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --termination_evaluation_frequency ${termination_eval_freq} \
        --step_size_policy constant \
        --restart_scheme ${PDLP_restart_scheme} \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --save_convergence_data ${save_convergence_data} \
        --save_detailed_convergence_data ${save_detailed} \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency} \
        --dwifob_option ${dwifob_option}
        
elif [ "$solver" == "+primal_weight" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --termination_evaluation_frequency ${termination_eval_freq} \
        --step_size_policy constant \
        --restart_scheme ${PDLP_restart_scheme} \
        --save_convergence_data ${save_convergence_data} \
        --save_detailed_convergence_data ${save_detailed} \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency} \
        --dwifob_option ${dwifob_option}

elif [ "$solver" == "+step_size" ]; then
   julia --project=scripts scripts/test_solver.jl \
        --instance_path $instance_path \
        --output_dir $output_file_base \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --termination_evaluation_frequency ${termination_eval_freq} \
        --restart_scheme ${PDLP_restart_scheme} \
        --save_convergence_data ${save_convergence_data} \
        --save_detailed_convergence_data ${save_detailed} \
        --steering_vectors ${use_steering} \
        --max_memory "${max_memory_input}" \
        --fast_dwifob ${use_fast} \
        --dwifob_restart ${restart_scheme} \
        --dwifob_restart_frequency ${restart_frequency} \
        --dwifob_option ${dwifob_option}
        
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
