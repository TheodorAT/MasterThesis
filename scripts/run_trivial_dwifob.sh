# Script for testing the DWIFOB solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"

# The selected solver: (either pdhg or dwifob)
solver="dwifob"
use_steering="true"
instance_path=./test/trivial_lp_model.mps
experiment_name="trivial_test_${solver}_${tolerance}"
output_file_base="./results/${experiment_name}"

declare -a max_memory_list=(3) 

# Bulding a string for the max memory input to the julia program: 
max_memory_input="["
for max_memory in "${max_memory_list[@]}" 
do
  max_memory_input="${max_memory_input}${max_memory}, "
done
max_memory_input="${max_memory_input::-2}]"

echo "Solving lp with ${solver}..."

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
        --fast_dwifob true \
        --dwifob_option "alt_A"

# Removing the temporary files:
for max_memory in "${max_memory_list[@]}" 
do
  log_dir_name_suffix="_m=${max_memory}"
  rm -rf ${output_file_base}${log_dir_name_suffix}
done
