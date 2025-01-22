# Script to run the lp benchmarking (without presolving). 

# This is the solver to be used: Acceptable values are:
#   pdhg (which uses vanilla pdhg)
#   pdhg-steering (which uses vanilla pdhg combined with steering vectors)
#   pdlp (which uses pdhg including optimizations made in the google research papers)
#   scs (which uses scs, a commercial solver)
solver="$1" 

# This is the error tolerance to be used in the solver
tolerance="$2"

declare -a instances=("nug08-3rd") # When actually doing real measurements, add more instances here.

if [ "$solver" == "pdhg" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdhg..."

    # Add the settings to make the solver remove the optimizations made in the google paper.
    julia --project=scripts scripts/solve_qp.jl \
    --instance_path ${HOME}/lp_benchmark/${INSTANCE}.mps.gz --method "pdhg" \
    --output_dir ./tmp/${solver}_solve_${tolerance} \
    --relative_optimality_tol ${tolerance} --absolute_optimality_tol ${tolerance} \
    --restart_scheme "no_restart" \
    --l_inf_ruiz_iterations 0 \
    --pock_chambolle_rescaling false \
    --scale_invariant_initial_primal_weight false \
    --step_size_policy "constant"
  done
elif [ "$solver" == "pdhg-steering" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdhg-steering..."

    # Add the settings to make the solver remove the optimizations made in the google paper.
    julia --project=scripts scripts/solve_qp.jl \
    --instance_path ${HOME}/lp_benchmark/${INSTANCE}.mps.gz --method "pdhg" \
    --output_dir ./tmp/${solver}_solve_${tolerance} \
    --relative_optimality_tol ${tolerance} --absolute_optimality_tol ${tolerance} \
    --restart_scheme "no_restart" \
    --l_inf_ruiz_iterations 0 \
    --pock_chambolle_rescaling false \
    --scale_invariant_initial_primal_weight false \
    --step_size_policy "constant" \
    --steering_vectors true
  done
elif [ "$solver" == "pdlp" ]; then
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with pdlp..."

    # Calling the solver with the default settings, which include the optimization techniques from the google research papers.
    julia --project=scripts scripts/solve_qp.jl \
    --instance_path ${HOME}/lp_benchmark/${INSTANCE}.mps.gz --method "pdhg" \
    --output_dir ./tmp/${solver}_solve_${tolerance} \
    --relative_optimality_tol ${tolerance} --absolute_optimality_tol ${tolerance}
  done
elif [ "$solver" == "scs" ]; then  
  for INSTANCE in "${instances[@]}" 
  do
    echo "Solving ${INSTANCE} with scs..."

    julia --project=scripts scripts/solve_lp_external.jl \
      --instance_path ${HOME}/lp_benchmark/${INSTANCE}.mps.gz --solver scs-indirect \
      --output_dir ./tmp/${solver}_solve_${tolerance} --tolerance ${tolerance}
  done
else
  echo "Invalid solver used"
  return
fi

echo "All problems solved, storing data in csv format..."

## Storing the results in CSV file using the process_json_to_csv.jl script:
echo '{"datasets": [
   {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'"}, "logs_directory": "./tmp/'${solver}'_solve_'${tolerance}'"}
], "config_labels": ["solver", "tolerance"]}' > ./tmp/layout.json

julia --project=benchmarking benchmarking/process_json_to_csv.jl ./tmp/layout.json ./tmp/lp_benchmark_${solver}_${tolerance}.csv

rm ./tmp/layout.json

echo "If no errors above then script finished successfully"
