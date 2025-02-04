# Script for testing the DWIFOB solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"

# The selected solver: (either pdhg or dwifob)
solver="dwifob"

instance_path=./test/trivial_lp_model.mps
experiment_name="trivial_test_${solver}_${tolerance}"

# INSTANCE="nug08-3rd"
# instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
# experiment_name="${INSTANCE}_test_power_step_${solver}_${tolerance}"

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