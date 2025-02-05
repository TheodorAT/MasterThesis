# Script for testing the PDHG solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"

# The selected solver: (always pdhg from this script)
# Select from "pdhg", "+restarts", "+scaling", "primal_weight", "+step_size"
solver="pdhg"
use_steering="false"

# INSTANCE="trivial_lp_model"
# instance_path=./test/${INSTANCE}.mps
# experiment_name="trivial_test_fast_${solver}_${tolerance}"

INSTANCE="nug08-3rd"
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
experiment_name="${INSTANCE}_test_${solver}_${tolerance}_PDLPscaling"

output_dir_base="./results/${INSTANCE}"
output_dir="${output_dir_base}/${solver}_solve_${INSTANCE}_${tolerance}"


echo "Solving ${INSTANCE} with ${solver}..."

if [ "$solver" == "pdhg" ]; then # This is the baseline vanilla PDHG:   
  julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --restart_scheme "no_restart" \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false

elif [ "$solver" == "+restarts" ]; then
   julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --l_inf_ruiz_iterations 0 \
        --pock_chambolle_rescaling false \
        --l2_norm_rescaling false \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false

elif [ "$solver" == "+scaling" ]; then
   julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant \
        --primal_weight_update_smoothing 0.0 \
        --scale_invariant_initial_primal_weight false  
        
elif [ "$solver" == "+primal_weight" ]; then
   julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 \
        --step_size_policy constant

elif [ "$solver" == "+step_size" ]; then
   julia --project=scripts scripts/solve_qp.jl \
        --instance_path $instance_path \
        --output_dir $output_dir \
        --method "pdhg" \
        --relative_optimality_tol ${tolerance} \
        --absolute_optimality_tol ${tolerance} \
        --iteration_limit 5000 
fi

echo "Problems solved, storing data in csv format..."

# Creating the JSON for collecting the results using another Julia Script:
echo '{"datasets": [
   {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'"}, 
   "logs_directory": "'${output_dir}'"}
], "config_labels": ["solver", "tolerance"]}' > ./results/layout.json

## Storing the results in CSV file using the process_json_to_csv.jl script:
julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv

# Removing the temporary files:
rm ./results/layout.json
rm -rf $output_dir

echo "Done"
