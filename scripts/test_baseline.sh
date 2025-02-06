# Script for testing the PDHG solver: 

# This is the error tolerance to be used in the solver:
tolerance="1e-4"
# INSTANCE="trivial_lp_model"
INSTANCE="nug08-3rd"

### Below this point are no settings ####
instance_path=${HOME}/lp_benchmark/${INSTANCE}.mps.gz
base_experiment_name="${INSTANCE}_test_pdhg_variants_${tolerance}"

# The selected solver:
declare -a solver_list=("pdhg" "+restarts" "+scaling" "+primal_weight" "+step_size") 

json_content='{"datasets": ['

for solver in "${solver_list[@]}"
do
   experiment_name="${INSTANCE}_test_${solver}_${tolerance}"
   output_dir="./results/${experiment_name}"

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
   
   json_content=${json_content}$'
      {"config": {"solver": "'${solver}'", "tolerance": "'${tolerance}'"},
         "logs_directory": "'${output_dir}'"},'
done

json_content=${json_content::-1}' 
], "config_labels": ["solver", "tolerance"]}'

echo "$json_content" > ./results/layout.json

echo "Problems solved, storing data in csv format..."
## Storing the results in CSV file using the process_json_to_csv.jl script:
julia --project=benchmarking benchmarking/process_json_to_csv.jl ./results/layout.json ./results/${experiment_name}.csv

# Removing the temporary files:
for solver in "${solver_list[@]}"
do
   experiment_name="${INSTANCE}_test_${solver}_${tolerance}"
   output_dir="./results/${experiment_name}"
   rm -rf $output_dir
done
rm ./results/layout.json

echo "Done"
