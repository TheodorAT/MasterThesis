# Script to run the lp benchmarking (without presolving). 

# This is the solver to be used: Acceptable values are:
#   pdhg (which uses vanilla pdhg)
#   dwifob (which uses vanilla dwifob)
#   pdhg+primal (which uses pdhg including optimizations up to primal weight update.)
#   dwifob+primal (uses dwifob + PDLP optimizations up to primal weight update.)
#   pdlp (which uses pdhg including optimizations made in the google research papers)
#   dwifob+step_size (uses dwifob + PDLP optimizations up to dynamic step size.)
#   scs (which uses scs, a free to use solver in Julia)
solver="dwifob+primal"
tolerance="1e-4"        # This is the error tolerance to be used in the solver.
iteration_limit="10000" # Iteration limit for the test run. 
dwifob_option="alt_C"           # Chose between "alt_A", "alt_B", "alt_C", anything else means the original.

# Get a list of all instances:
declare -a instances=() 
while IFS= read -r line; do
  if [[ $line != \#* ]]; then # Ignore commented lines starting with #.
    instances+=("${line//[$'\t\r\n ']}") # Here we remove the newlines and blank characters from the variable
  fi
done < "./benchmarking/lp_benchmark_instance_list"

save_convergence_data="true"
max_memory_input="[1]"

# declare -a instances=("nug08-3rd") # For testing the script
experiment_name="fast_lp_benchmark_${solver}_${dwifob_option}_m=${max_memory_input}_${tolerance}"

# Below are no more settings:
output_dir="./results/${experiment_name}"

if [ "$solver" == "pdhg" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdhg..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    julia --project=scripts scripts/solve_qp.jl \
         --instance_path $instance_path \
         --output_dir $output_dir \
         --method "pdhg" \
         --relative_optimality_tol ${tolerance} \
         --absolute_optimality_tol ${tolerance} \
         --iteration_limit $iteration_limit \
         --step_size_policy constant \
         --l_inf_ruiz_iterations 0 \
         --pock_chambolle_rescaling false \
         --l2_norm_rescaling false \
         --restart_scheme "no_restart" \
         --primal_weight_update_smoothing 0.0 \
         --scale_invariant_initial_primal_weight false \
         --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "pdhg+primal" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdhg+primal..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --step_size_policy "constant" \
        --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "pdlp" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdlp..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    # Calling the solver with the default settings, which include the optimization techniques from the google research papers.
    julia --project=scripts scripts/solve_qp.jl \
         --instance_path $instance_path \
         --output_dir $output_dir \
         --method "pdhg" \
         --relative_optimality_tol ${tolerance} \
         --absolute_optimality_tol ${tolerance} \
         --iteration_limit $iteration_limit \
         --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "dwifob" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with dwifob..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    # Add the settings to make the solver remove the optimizations made in the google paper, leaving only the pure PDHG method.
    # Additionally, we add the argument for using steering vectors.
    julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --step_size_policy "constant" \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --restart_scheme "no_restart" \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false \
        --steering_vectors true \
        --fast_dwifob true \
        --dwifob_option ${dwifob_option} \
        --max_memory "${max_memory_input}" \
        --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "dwifob+primal" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with dwifob+primal..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --step_size_policy "constant" \
        --steering_vectors true \
        --max_memory "${max_memory_input}" \
        --fast_dwifob true \
        --dwifob_option ${dwifob_option} \
        --dwifob_restart "constant" \
        --dwifob_restart_frequency 40 \
        --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "dwifob+step_size" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with dwifob+step_size..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

        julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit $iteration_limit \
        --steering_vectors true \
        --max_memory "${max_memory_input}" \
        --fast_dwifob true \
        --dwifob_option ${dwifob_option} \
        --dwifob_restart "constant" \
        --dwifob_restart_frequency 40 \
        --save_convergence_data ${save_convergence_data}

  done
elif [ "$solver" == "scs" ]; then  
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with scs..."
    instance_path="${HOME}/lp_benchmark/${INSTANCE}.mps.gz"

    julia --project=scripts scripts/solve_lp_external.jl \
      --instance_path $instance_path \
      --output_dir $output_dir --tolerance ${tolerance}

  done
fi

echo "All problems solved, storing data in csv format..."

## Storing the results in CSV file using the process_json_to_csv.jl script:
echo '{"datasets": [
   {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'"}, "logs_directory": "'${output_dir}'"}
], "config_labels": ["solver", "tolerance"]}' > ./results/layout.json

julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv

# DO NOT REMOVE THE OUTPUT DIR, THOSE RESULTS TAKE HOURS TO GET!!!
rm ./results/layout.json

echo "If no errors above then script finished successfully"