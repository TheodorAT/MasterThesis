# Copyright 2021 The FirstOrderLp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an interface to FirstOrderLp for solving a single QP or LP and writing
# the solution and solve statistics to a file. Run with --help for a pretty
# description of the arguments. See the comments on solve_instance_and_output
# for a description of the output formats.

import FirstOrderLp
include("input_output.jl")

function main()
  parsed_args = parse_command_line()

  if parsed_args["method"] == "mirror-prox" || parsed_args["method"] == "pdhg"
    restart_params = FirstOrderLp.construct_restart_parameters(
      string_to_restart_scheme(parsed_args["restart_scheme"]),
      string_to_restart_to_current_metric(
        parsed_args["restart_to_current_metric"],
      ),
      parsed_args["restart_frequency"],
      parsed_args["artificial_restart_threshold"],
      parsed_args["sufficient_reduction_for_restart"],
      parsed_args["necessary_reduction_for_restart"],
      parsed_args["primal_weight_update_smoothing"],
      parsed_args["use_approximate_localized_duality_gap"],
    )

    pock_chambolle_alpha = nothing
    if parsed_args["pock_chambolle_rescaling"]
      pock_chambolle_alpha = parsed_args["pock_chambolle_alpha"]
    end

    termination_criteria = FirstOrderLp.construct_termination_criteria()
    if parsed_args["optimality_norm"] == "l2"
      termination_criteria.optimality_norm = FirstOrderLp.OptimalityNorm.L2
    elseif parsed_args["optimality_norm"] == "l_inf"
      termination_criteria.optimality_norm = FirstOrderLp.OptimalityNorm.L_INF
    elseif parsed_args["optimality_norm"] !== nothing
      error("Unknown termination norm.")
    end
    for (field_name, arg_name) in [
      (:eps_optimal_absolute, "absolute_optimality_tol"),
      (:eps_optimal_relative, "relative_optimality_tol"),
      (:eps_primal_infeasible, "eps_primal_infeasible"),
      (:eps_dual_infeasible, "eps_dual_infeasible"),
      (:time_sec_limit, "time_sec_limit"),
      (:iteration_limit, "iteration_limit"),
      (:kkt_matrix_pass_limit, "kkt_matrix_pass_limit"),
    ]
      if parsed_args[arg_name] !== nothing
        setproperty!(termination_criteria, field_name, parsed_args[arg_name])
      end
    end

    if parsed_args["method"] == "mirror-prox"
      parameters = FirstOrderLp.MirrorProxParameters(
        parsed_args["l_inf_ruiz_iterations"],
        parsed_args["l2_norm_rescaling"],
        pock_chambolle_alpha,
        parsed_args["primal_importance"],
        parsed_args["scale_invariant_initial_primal_weight"],
        parsed_args["diagonal_scaling"],
        parsed_args["verbosity"],
        parsed_args["record_iteration_stats"],
        parsed_args["termination_evaluation_frequency"],
        termination_criteria,
        restart_params,
      )
    elseif parsed_args["method"] == "pdhg"
      if parsed_args["step_size_policy"] == "malitsky-pock"
        step_size_policy_params = FirstOrderLp.MalitskyPockStepsizeParameters(
          parsed_args["malitsky_pock_downscaling_factor"],
          parsed_args["malitsky_pock_breaking_factor"],
          parsed_args["malitsky_pock_interpolation_coefficient"],
        )
      elseif parsed_args["step_size_policy"] == "constant"
        step_size_policy_params = FirstOrderLp.ConstantStepsizeParams()
      else
        step_size_policy_params = FirstOrderLp.AdaptiveStepsizeParams(
          parsed_args["adaptive_step_size_reduction_exponent"],
          parsed_args["adaptive_step_size_growth_exponent"],
        )
      end
      parameters = FirstOrderLp.PdhgParameters(
        parsed_args["l_inf_ruiz_iterations"],
        parsed_args["l2_norm_rescaling"],
        pock_chambolle_alpha,
        parsed_args["primal_importance"],
        parsed_args["scale_invariant_initial_primal_weight"],
        parsed_args["verbosity"],
        parsed_args["record_iteration_stats"],
        parsed_args["termination_evaluation_frequency"],
        termination_criteria,
        restart_params,
        step_size_policy_params,
        parsed_args["steering_vectors"],
      )
    end
  else
    error("`method` arg must be either `mirror-prox` or `pdhg`.")
  end

  solve_instance_and_output(
    parameters,
    parsed_args["output_dir"],
    parsed_args["instance_path"],
    parsed_args["redirect_stdio"],
    parsed_args["transform_bounds_into_linear_constraints"],
    parsed_args["fixed_format_input"],
  )
end

main()
