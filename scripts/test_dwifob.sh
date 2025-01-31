# Script for testing the DWIFOB solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"

# The selected solver: (either pdhg or dwifob)
solver="dwifob"

if [ "$solver" == "pdhg" ]; then
  use_steering="false"
elif [ "$solver" == "dwifob" ]; then
  use_steering="true"
else
  echo "Invalid solver used"
  return
fi

echo "Solving trivial lp with ${solver}..."

julia --project=scripts scripts/test_solver.jl \
        --instance_path ./test/trivial_lp_model.mps \
        --output_dir ./results/${solver}_solve_trivial_${tolerance} \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --restart_scheme "no_restart" \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --scale_invariant_initial_primal_weight false \
        --step_size_policy "constant" \
        --steering_vectors ${use_steering}

# echo "Problems solved, storing data in csv format..."

# ## Storing the results in CSV file using the process_json_to_csv.jl script:
# echo '{"datasets": [
#    {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'"}, "logs_directory": "./results/'${solver}'_solve_trivial_'${tolerance}'"}
# ], "config_labels": ["solver", "tolerance"]}' > ./results/layout.json

# julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/trivial_test_${solver}_${tolerance}.csv

# rm ./results/layout.json
# rm -rf ./results/${solver}_solve_trivial_${tolerance}

# echo "Done"
