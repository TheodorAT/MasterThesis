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

using LinearAlgebra
using JSON3

"""
Parameters of the Malitsky and Pock lineseach algorithm
(https://arxiv.org/pdf/1608.08883.pdf).
 """
struct MalitskyPockStepsizeParameters

  """
  Factor by which the step size is multiplied for in the inner loop.
  Valid values: interval (0, 1). Corresponds to mu in the paper.
  """
  downscaling_factor::Float64

  # Y. Malitsky notes that while the theory requires the value to be strictly
  # less than 1, a value of 1 should work fine in practice.
  """
  Breaking factor that defines the stopping criteria of the linesearch.
  Valid values: interval (0, 1]. Corresponds to delta in the paper.
  """
  breaking_factor::Float64

  """
  Interpolation coefficient to pick next step size. The next step size can be
  picked within an interval [a, b] (See Step 2 of Algorithm 1). The solver uses
  a + interpolation_coefficient * (b - a). Valid values: interval [0, 1].
  """
  interpolation_coefficient::Float64
end

"""
Parameters used for the adaptive stepsize policy.

At each inner iteration we update the step size as follows
Our step sizes are a factor
 1 - (iteration + 1)^(-reduction_exponent)
smaller than they could be as a margin to reduce rejected steps.
From the first term when we have to reject a step, the step_size
decreases by a factor of at least 1 - (iteration + 1)^(-reduction_exponent).
From the second term we increase the step_size by a factor of at most
1 + (iteration + 1)^(-growth_exponent)
Therefore if more than order
(iteration + 1)^(reduction_exponent - growth_exponent)
fraction of the iterations have a rejected step we overall decrease the
step_size. When the step_size is below the inverse of the max singular
value we stop having rejected steps.
"""
struct AdaptiveStepsizeParams
  reduction_exponent::Float64
  growth_exponent::Float64
end

"""
Empty placeholder for the parameters of a constant step size policy.
"""
struct ConstantStepsizeParams end

"""
A PdhgParameters struct specifies the parameters for solving the saddle
point formulation of an problem using primal-dual hybrid gradient.
Quadratic Programming Problem (see quadratic_programming.jl):
minimize 1/2 * x' * objective_matrix * x + objective_vector' * x
         + objective_constant
s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]
     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end, :]
     variable_lower_bound <= x <= variable_upper_bound
We use notation from Chambolle and Pock, "On the ergodic convergence rates of a
first-order primal-dual algorithm"
(http://www.optimization-online.org/DB_FILE/2014/09/4532.pdf).
That paper doesn't explicitly use the terminology "primal-dual hybrid gradient"
but their Theorem 1 is analyzing PDHG. In this file "Theorem 1" without further
reference refers to that paper.
Our problem is equivalent to the saddle point problem:
    min_x max_y L(x, y)
where
    L(x, y) = y' K x + f(x) + g(x) - h*(y)
    K = -constraint_matrix
    f(x) = objective_constant + objective_vector' x + 1/2*x' objective_matrix x
    g(x) = 0 if variable_lower_bound <= x <= variable_upper_bound
           otherwise infinity
    h*(y) = -right_hand_side' y if y[(num_equalities + 1):end] >= 0
                                otherwise infinity
Note that the places where g(x) and h*(y) are infinite effectively limits the
domain of the min and max. Therefore there's no infinity in the code.

The code uses slightly different notation from the Chambolle and Pock paper.
The primal and dual step sizes are parameterized as:
  tau = primal_step_size = step_size / primal_weight
  sigma = dual_step_size = step_size * primal_weight.
The primal_weight factor is named as such because this parameterization is
equivalent to defining the Bregman divergences as:
D_x(x, x bar) = 0.5 * primal_weight ||x - x bar||_2^2, and
D_y(y, y bar) = 0.5 / primal_weight ||y - y bar||_2^2.

The parameter primal_weight is adjusted smoothly at each restart; to balance the
primal and dual distances traveled since the last restart; see
compute_new_primal_weight().

The adaptive rule adjusts step_size to be as large as possible without
violating the condition assumed in Theorem 1. Adjusting the step size
unfortunately seems to invalidate that Theorem (unlike the case of mirror prox)
but this step size adjustment heuristic seems to work fine in practice.
See comments in the code for details.

TODO: compare the above step size scheme with the scheme by Goldstein
et al (https://arxiv.org/pdf/1305.0546.pdf).

TODO: explore PDHG variants with tuning parameters, e.g. the
overrelaxed and intertial variants in Chambolle and Pock and the algorithm in
"An Algorithmic Framework of Generalized Primal-Dual Hybrid Gradient Methods for
Saddle Point Problems" by Bingsheng He, Feng Ma, Xiaoming Yuan
(http://www.optimization-online.org/DB_FILE/2016/02/5315.pdf).
"""
struct PdhgParameters
  """
  Number of L_infinity Ruiz rescaling iterations to apply to the constraint
  matrix. Zero disables this rescaling pass.
  """
  l_inf_ruiz_iterations::Int

  """
  If true, applies L2 norm rescaling after the Ruiz rescaling.
  """
  l2_norm_rescaling::Bool

  """
  If not `nothing`, runs Pock-Chambolle rescaling with the given alpha exponent
  parameter.
  """
  pock_chambolle_alpha::Union{Float64,Nothing}

  """
  Used to bias the initial value of the primal/dual balancing parameter
  primal_weight. Must be positive. See also
  scale_invariant_initial_primal_weight.
  """
  primal_importance::Float64

  """
  If true, computes the initial primal weight with a scale-invariant formula
  biased by primal_importance; see select_initial_primal_weight() for more
  details. If false, primal_importance itself is used as the initial primal
  weight.
  """
  scale_invariant_initial_primal_weight::Bool

  """
  If >= 4 a line of debugging info is printed during some iterations. If >= 2
  some info is printed about the final solution.
  """
  verbosity::Int64

  """
  Whether to record an IterationStats object. If false, only iteration stats
  for the final (terminating) iteration are recorded.
  """
  record_iteration_stats::Bool

  """
  Check for termination with this frequency (in iterations).
  """
  termination_evaluation_frequency::Int32

  """
  The termination criteria for the algorithm.
  """
  termination_criteria::TerminationCriteria

  """
  Parameters that control when the algorithm restarts and whether it resets
  to the average or the current iterate. Also, controls the primal weight
  updates.
  """
  restart_params::RestartParameters

  """
  Parameters of the step size policy. There are three step size policies
  implemented: Adaptive, Malitsky and Pock, and constant step size.
  """
  step_size_policy_params::Union{
    MalitskyPockStepsizeParameters,
    AdaptiveStepsizeParams,
    ConstantStepsizeParams,
  }

  """ Whether or not to save the last iterates in a json file or not"""
  save_solution_json::Bool

  """ Whether or not the program should save the convergence stats in a json file or not"""
  should_save_convergence_data::Bool

  """ Whether or not the program should save the detailed convergence stats in a json file or not"""
  should_save_detailed_data::Bool

  """
    If true, applies steering vectors into the solver algorithm.
  """
  steering_vectors::Bool

  """ 
    If true, applies the faster version of DWIFOB in the solver algorithm 
  """
  fast_dwifob::Bool

  """
    Decides the dwifob option. 
    (The different versions have different input to the RAA, or take inspiration from the NOFOB article)
  """
  dwifob_option::String 

  """
    Dwifob restart scheme
  """
  dwifob_restart::String

  """
    Dwifob restart frequency used for the constant dwifob_restart scheme.  
  """
  dwifob_restart_frequency::Int64
end

"""
A PdhgSolverState struct specifies the state of the solver.  It is used to
pass information among the main solver function and other helper functions.
"""
mutable struct PdhgSolverState
  current_primal_solution::Vector{Float64}

  current_dual_solution::Vector{Float64}

  """
  Current primal delta. That is current_primal_solution - previous_primal_solution.
  """
  delta_primal::Vector{Float64}

  """
  Current dual delta. That is current_dual_solution - previous_dual_solution.
  """
  delta_dual::Vector{Float64}

  """
  A cache of constraint_matrix' * current_dual_solution.
  """
  current_dual_product::Vector{Float64}

  solution_weighted_avg::SolutionWeightedAverage

  step_size::Float64

  primal_weight::Float64

  """
  True only if the solver was unable to take a step in the previous
  iterations because of numerical issues, and must terminate on the next step.
  """
  numerical_error::Bool

  """
  Number of KKT passes so far.
  """
  cumulative_kkt_passes::Float64

  """
  Total number of iterations. This includes inner iterations.
  """
  total_number_iterations::Int64

  """
  Latest required_ratio. This field is only used with the adaptive step size.
  The proof of Theorem 1 requires 1 >= required_ratio.
  """
  required_ratio::Union{Float64,Nothing}

  """
  Ratio between the last two step sizes: step_size(n)/step_size(n-1).
  It is only saved while using Malitsky and Pock linesearch.
  """
  ratio_step_sizes::Union{Float64,Nothing}
end

struct DwifobParameters
  max_memory::Int64
end

mutable struct DwifobSolverState
  """ How many terms back in time we use to compute the deviations. """
  max_memory::Int64

  """ The number of iterations since last restart. """
  current_iteration::Int64 

  lambda_k::Float64
  lambda_next::Float64

  zeta_k::Float64
  epsilon::Float64

  """
    The maximum singular value of the K matrix, this is used to limit the step size.  
  """
  maximum_singular_value::Float64

  """
    The last m_n iterates of the primal, used in the RAA in DWIFOB.
    Stores x_{n}, x_{n-1}, ... , x_{n-m_n}
  """
  primal_iterates::Vector{Vector{Float64}}

  """
    The last m_n iterates of the dual, used in the RAA in DWIFOB.
    Stores my_{n}, my_{n-1}, ... , my_{n-m_n}
  """
  dual_iterates::Vector{Vector{Float64}}

  """
    The last m_n iterates of the primal^{hat}, used in the RAA in DWIFOB.
    Stores x^{hat}_{n}, x^{hat}_{n-1}, ... , x^{hat}_{n-m_n}
  """
  x_hat_iterates::Vector{Vector{Float64}}

  """
    The last m_n iterates of the dual^{hat}, used in the RAA in DWIFOB.
    Stores my^{hat}_{n}, my^{hat}_{n-1}, ... , my^{hat}_{n-m_n}
  """
  y_hat_iterates::Vector{Vector{Float64}}

  current_primal_deviation::Vector{Float64}
  current_dual_deviation::Vector{Float64}

  current_u_hat_x_deviation_sum::Vector{Float64}
  current_u_hat_y_deviation_sum::Vector{Float64}

end

# The following parameters are for the more efficient version of DWIFOB steps:
mutable struct DwifobMatrixCache
  """ Storing the history of cached values for K * x_k """
  K_x_iterates::Vector{Vector{Float64}}  
  """ Storing the history of cached values for K^T * y_k """
  K_trans_y_iterates::Vector{Vector{Float64}}

  """ Storing the current value of K * x """
  K_x_current::Vector{Float64}
  """ Storing the current value of K^T * y """
  K_trans_y_current::Vector{Float64}

  """ Storing the current value of K * x_hat """
  K_x_hat_current::Vector{Float64}
  """ Storing the current value of K^T * y_hat """
  K_trans_y_hat_current::Vector{Float64}

  # The deviations: 
  """ Storing the current value of K * u_x_k """
  K_u_x_current::Vector{Float64}
end

"""
Defines the primal norm and dual norm using the norms of matrices, step_size
and primal_weight. This is used only for interacting with general utilities
in saddle_point.jl. PDHG utilities these norms implicitly.
"""
function define_norms(
  primal_size::Int64,
  dual_size::Int64,
  step_size::Float64,
  primal_weight::Float64,
)
  # TODO: Should these norms include the step size?
  primal_norm_params = 1 / step_size * primal_weight * ones(primal_size)
  dual_norm_params = 1 / step_size / primal_weight * ones(dual_size)

  return primal_norm_params, dual_norm_params
end

"""
Logging while the algorithm is running.
"""
function pdhg_specific_log(
  problem::QuadraticProgrammingProblem,
  iteration::Int64,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  step_size::Float64,
  required_ratio::Union{Float64,Nothing},
  primal_weight::Float64,
)
  Printf.@printf(
    "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
    iteration,
    norm(current_primal_solution),
    norm(current_dual_solution),
    1 / step_size,
  )
  if !isnothing(required_ratio)
    Printf.@printf(
      "   primal_weight=%18g dual_obj=%18g  inverse_ss=%18g\n",
      primal_weight,
      corrected_dual_obj(
        problem,
        current_primal_solution,
        current_dual_solution,
      ),
      required_ratio
    )
  else
    Printf.@printf(
      "   primal_weight=%18g dual_obj=%18g\n",
      primal_weight,
      corrected_dual_obj(
        problem,
        current_primal_solution,
        current_dual_solution,
      )
    )
  end
end

"""
Logging for when the algorithm terminates.
"""
function pdhg_final_log(
  problem::QuadraticProgrammingProblem,
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  verbosity::Int64,
  iteration::Int64,
  termination_reason::TerminationReason,
  last_iteration_stats::IterationStats,
)

  if verbosity >= 2
    infeas = max_primal_violation(problem, avg_primal_solution)
    primal_obj_val = primal_obj(problem, avg_primal_solution)
    dual_stats =
      compute_dual_stats(problem, avg_primal_solution, avg_dual_solution)
    println("Avg solution:")
    Printf.@printf(
      "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
      infeas,
      primal_obj_val,
      norm(dual_stats.dual_residual, Inf),
      dual_stats.dual_objective
    )
    Printf.@printf(
      "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
      norm(avg_primal_solution, 1),
      norm(avg_primal_solution),
      norm(avg_primal_solution, Inf)
    )
    Printf.@printf(
      "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
      norm(avg_dual_solution, 1),
      norm(avg_dual_solution),
      norm(avg_dual_solution, Inf)
    )
  end

  generic_final_log(
    problem,
    avg_primal_solution,
    avg_dual_solution,
    last_iteration_stats,
    verbosity,
    iteration,
    termination_reason,
  )
end

"""
Estimate the probability that the power method, after k iterations, has relative
error > epsilon.  This is based on Theorem 4.1(a) (on page 13) from
"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf
"""
function power_method_failure_probability(
  dimension::Int64,
  epsilon::Float64,
  k::Int64,
)
  if k < 2 || epsilon <= 0.0
    # The theorem requires epsilon > 0 and k >= 2.
    return 1.0
  end
  return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) *
         sqrt(dimension) *
         (1.0 - epsilon)^(k - 1 / 2)
end

"""
Estimate the maximum singular value using power method
https://en.wikipedia.org/wiki/Power_iteration, returning a result with
desired_relative_error with probability at least 1 - probability_of_failure.

Note that this will take approximately log(n / delta^2)/(2 * epsilon) iterations
as per the discussion at the bottom of page 15 of

"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf

For lighter reading on this topic see
https://courses.cs.washington.edu/courses/cse521/16sp/521-lecture-13.pdf
which does not include the failure probability.

# Output
A tuple containing:
- estimate of the maximum singular value
- the number of power iterations required to compute it
"""
function estimate_maximum_singular_value(
  matrix::SparseMatrixCSC{Float64,Int64};
  probability_of_failure = 0.01::Float64,
  desired_relative_error = 0.1::Float64,
  seed::Int64 = 1,
)
  # Epsilon is the relative error on the eigenvalue of matrix' * matrix.
  epsilon = 1.0 - (1.0 - desired_relative_error)^2
  # Use the power method on matrix' * matrix
  x = randn(Random.MersenneTwister(seed), size(matrix, 2))

  number_of_power_iterations = 0
  while power_method_failure_probability(
    size(matrix, 2),
    epsilon,
    number_of_power_iterations,
  ) > probability_of_failure
    x = x / norm(x, 2)
    x = matrix' * (matrix * x)
    number_of_power_iterations += 1
  end

  # The singular value is the square root of the maximum eigenvalue of
  # matrix' * matrix
  return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
  number_of_power_iterations
end

function compute_next_primal_solution(
  problem::QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_product::Vector{Float64},
  step_size::Float64,
  primal_weight::Float64,
)
  # The next lines compute the primal portion of the PDHG algorithm:
  # argmin_x [gradient(f)(current_primal_solution)'x + g(x)
  #          + current_dual_solution' K x
  #          + (1 / step_size) * D_x(x, current_primal_solution)]
  # See Sections 2-3 of Chambolle and Pock and the comment above
  # PdhgParameters.
  # This minimization is easy to do in closed form since it can be separated
  # into independent problems for each of the primal variables. The
  # projection onto the primal feasibility set comes from the closed form
  # for the above minimization and the cases where g(x) is infinite - there
  # isn't officially any projection step in the algorithm.
  primal_gradient = compute_primal_gradient_from_dual_product(
    problem,
    current_primal_solution,
    current_dual_product,
  )

  next_primal =
    current_primal_solution .- (step_size / primal_weight) .* primal_gradient
  project_primal!(next_primal, problem)
  return next_primal
end

function compute_next_dual_solution(
  problem::QuadraticProgrammingProblem,
  current_primal_solution::Vector{Float64},
  next_primal::Vector{Float64},
  current_dual_solution::Vector{Float64},
  step_size::Float64,
  primal_weight::Float64;
  extrapolation_coefficient::Float64 = 1.0,
)
  # The next two lines compute the dual portion:
  # argmin_y [H*(y) - y' K (next_primal + extrapolation_coefficient*(next_primal - current_primal_solution)
  #           + 0.5*norm_Y(y-current_dual_solution)^2]
  dual_gradient = compute_dual_gradient(
    problem,
    next_primal .+
    extrapolation_coefficient .* (next_primal - current_primal_solution),
  )
  next_dual =
    current_dual_solution .+ (primal_weight * step_size) .* dual_gradient
  project_dual!(next_dual, problem)
  next_dual_product = problem.constraint_matrix' * next_dual
  return next_dual, next_dual_product
end

"""
Updates the solution fields of the solver state with the arguments given.
The function modifies the first argument: solver_state.
"""
function update_solution_in_solver_state(
  solver_state::PdhgSolverState,
  next_primal::Vector{Float64},
  next_dual::Vector{Float64},
  next_dual_product::Vector{Float64},
)
  solver_state.delta_primal = next_primal - solver_state.current_primal_solution
  solver_state.delta_dual = next_dual - solver_state.current_dual_solution
  solver_state.current_primal_solution = next_primal
  solver_state.current_dual_solution = next_dual
  solver_state.current_dual_product = next_dual_product

  weight = solver_state.step_size
  add_to_solution_weighted_average(
    solver_state.solution_weighted_avg,
    solver_state.current_primal_solution,
    solver_state.current_dual_solution,
    weight,
  )
end

"""
Computes the interaction and movement of the new iterates.
The movement is used to check if there is a numerical error (movement == 0.0)
and based on the theory (Theorem 1) the algorithm only moves if
interaction / movement <= step_size.
"""
function compute_interaction_and_movement(
  solver_state::PdhgSolverState,
  problem::QuadraticProgrammingProblem,
  next_primal::Vector{Float64},
  next_dual::Vector{Float64},
  next_dual_product::Vector{Float64},
)
  delta_primal = next_primal .- solver_state.current_primal_solution
  delta_dual = next_dual .- solver_state.current_dual_solution
  if iszero(problem.objective_matrix)
    primal_objective_interaction = 0.0
  else
    primal_objective_interaction =
      0.5 * (delta_primal' * problem.objective_matrix * delta_primal)
  end
  primal_dual_interaction =
    delta_primal' * (next_dual_product .- solver_state.current_dual_product)
  interaction = abs(primal_dual_interaction) + abs(primal_objective_interaction)
  movement =
    0.5 * solver_state.primal_weight * norm(delta_primal)^2 +
    (0.5 / solver_state.primal_weight) * norm(delta_dual)^2
  return interaction, movement
end

"""
Takes a step using Malitsky and Pock linesearch.
It modifies the third arguement: solver_state.
"""
function take_step(
  step_params::MalitskyPockStepsizeParameters,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
)
  if !is_linear_programming_problem(problem)
    error(
      "Malitsky and Pock linesearch is only supported for linear" *
      " programming problems.",
    )
  end

  step_size = solver_state.step_size
  ratio_step_sizes = solver_state.ratio_step_sizes
  done = false
  iter = 0

  next_primal = compute_next_primal_solution(
    problem,
    solver_state.current_primal_solution,
    solver_state.current_dual_product,
    step_size,
    solver_state.primal_weight,
  )
  solver_state.cumulative_kkt_passes += 0.5
  step_size =
    step_size +
    step_params.interpolation_coefficient *
    (sqrt(1 + ratio_step_sizes) - 1) *
    step_size

  max_iter = 60
  while !done && iter < max_iter
    iter += 1
    solver_state.total_number_iterations += 1
    ratio_step_sizes = step_size / solver_state.step_size

    # TODO: Get rid of the extra multiply by the constraint matrix (see Remark 1 in
    # https://arxiv.org/pdf/1608.08883.pdf)
    next_dual, next_dual_product = compute_next_dual_solution(
      problem,
      solver_state.current_primal_solution,
      next_primal,
      solver_state.current_dual_solution,
      step_size,
      solver_state.primal_weight;
      extrapolation_coefficient = ratio_step_sizes,
    )
    delta_dual = next_dual .- solver_state.current_dual_solution
    delta_dual_product = next_dual_product .- solver_state.current_dual_product
    # This is the ideal count. The current version of the code incurs in 1.0 kkt
    # pass. See TODO before the next_dual update above.
    solver_state.cumulative_kkt_passes += 0.5

    # The primal weight does not play a role in this condition. As noted in the
    # paper (See second paragraph of Section 2 in https://arxiv.org/pdf/1608.08883.pdf)
    # the coefficient on left-hand-side is equal to
    # sqrt(<primal_step_size> * <dual_step_size>) = step_size.
    # where the equality follows since the primal_weight in the primal and dual step
    # sizes cancel out.
    if step_size * norm(delta_dual_product) <=
       step_params.breaking_factor * norm(delta_dual)
      # Malitsky and Pock guarantee uses a nonsymmetric weighted average, the
      # primal variable average involves the initial point, while the dual
      # doesn't. See Theorem 2 in https://arxiv.org/pdf/1608.08883.pdf for
      # details.
      if solver_state.solution_weighted_avg.sum_primal_solutions_count == 0
        add_to_primal_solution_weighted_average(
          solver_state.solution_weighted_avg,
          solver_state.current_primal_solution,
          step_size * ratio_step_sizes,
        )
      end

      update_solution_in_solver_state(
        solver_state,
        next_primal,
        next_dual,
        next_dual_product,
      )
      done = true
    else
      step_size *= step_params.downscaling_factor
    end
  end
  if iter == max_iter && !done
    solver_state.numerical_error = true
    return
  end
  solver_state.step_size = step_size
  solver_state.ratio_step_sizes = ratio_step_sizes

end

"""
Takes a step using the adaptive step size.
It modifies the third argument: solver_state.
"""
function take_step(
  step_params::AdaptiveStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
)
  step_size = solver_state.step_size
  done = false
  iter = 0

  while !done
    iter += 1
    solver_state.total_number_iterations += 1

    next_primal = compute_next_primal_solution(
      problem,
      solver_state.current_primal_solution,
      solver_state.current_dual_product,
      step_size,
      solver_state.primal_weight,
    )

    next_dual, next_dual_product = compute_next_dual_solution(
      problem,
      solver_state.current_primal_solution,
      next_primal,
      solver_state.current_dual_solution,
      step_size,
      solver_state.primal_weight,
    )
    interaction, movement = compute_interaction_and_movement(
      solver_state,
      problem,
      next_primal,
      next_dual,
      next_dual_product,
    )
    solver_state.cumulative_kkt_passes += 1

    if movement == 0.0
      # The algorithm will terminate at the beginning of the next iteration
      solver_state.numerical_error = true
      break
    end
    # The proof of Theorem 1 requires movement / step_size >= interaction.
    if interaction > 0
      step_size_limit = movement / interaction
    else
      step_size_limit = Inf
    end

    if step_size <= step_size_limit
      update_solution_in_solver_state(
        solver_state,
        next_primal,
        next_dual,
        next_dual_product,
      )
      done = true
    end

    first_term = (step_size_limit * 
      (1 - (solver_state.total_number_iterations + 1)^(-step_params.reduction_exponent)))
    second_term = (step_size * 
      (1 + (solver_state.total_number_iterations + 1)^(-step_params.growth_exponent)))
    step_size = min(first_term, second_term)
  end
  solver_state.step_size = step_size
end

"""
Takes a step with constant step size.
It modifies the third argument: solver_state.
"""
function take_step(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
)
  next_primal = compute_next_primal_solution(
    problem,
    solver_state.current_primal_solution,
    solver_state.current_dual_product,
    solver_state.step_size,
    solver_state.primal_weight,
  )

  next_dual, next_dual_product = compute_next_dual_solution(
    problem,
    solver_state.current_primal_solution,
    next_primal,
    solver_state.current_dual_solution,
    solver_state.step_size,
    solver_state.primal_weight,
  )

  solver_state.cumulative_kkt_passes += 1

  update_solution_in_solver_state(
    solver_state,
    next_primal,
    next_dual,
    next_dual_product,
  )
end

"""
`optimize(params::PdhgParameters,
          original_problem::QuadraticProgrammingProblem)`

Solves a quadratic program using primal-dual hybrid gradient.

# Arguments
- `params::PdhgParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.

# Returns
A SaddlePointOutput struct containing the solution found.
"""
function optimize(
  params::PdhgParameters,
  original_problem::QuadraticProgrammingProblem,
  dwifob_params::Union{DwifobParameters, Nothing}=nothing,
  output_file = nothing
)
  validate(original_problem)
  qp_cache = cached_quadratic_program_info(original_problem)
  scaled_problem = rescale_problem(
    params.l_inf_ruiz_iterations,
    params.l2_norm_rescaling,
    params.pock_chambolle_alpha,
    params.verbosity,
    original_problem,
  )
  problem = scaled_problem.scaled_qp

  primal_size = length(problem.variable_lower_bound)
  dual_size = length(problem.right_hand_side)
  if params.primal_importance <= 0 || !isfinite(params.primal_importance)
    error("primal_importance must be positive and finite")
  end

  # TODO: Correctly account for the number of kkt passes in
  # initialization
  solver_state = PdhgSolverState(
    zeros(primal_size),  # current_primal_solution
    zeros(dual_size),    # current_dual_solution
    zeros(primal_size),  # delta_primal
    zeros(dual_size),    # delta_dual
    zeros(primal_size),  # current_dual_product
    initialize_solution_weighted_average(primal_size, dual_size),
    0.0,                 # step_size
    1.0,                 # primal_weight
    false,               # numerical_error
    0.0,                 # cumulative_kkt_passes
    0,                   # total_number_iterations
    nothing,             # required_ratio
    nothing,             # ratio_step_sizes
  )

  desired_relative_error = 0.2 
  maximum_singular_value, number_of_power_iterations =
    estimate_maximum_singular_value(
      problem.constraint_matrix,
      probability_of_failure = 0.001,
      desired_relative_error = desired_relative_error,
    )

  if params.step_size_policy_params isa AdaptiveStepsizeParams
    solver_state.cumulative_kkt_passes += 0.5
    solver_state.step_size = 1.0 / norm(problem.constraint_matrix, Inf) # FIXME: Why do they do this norm for adaptive step sizes?
  elseif params.step_size_policy_params isa MalitskyPockStepsizeParameters
    solver_state.cumulative_kkt_passes += 0.5
    solver_state.step_size = 1.0 / norm(problem.constraint_matrix, Inf)
    solver_state.ratio_step_sizes = 1.0
  else 
    # TODO: Can we increase performance by using values closer to 1? 0.99?
    solver_state.step_size = (1 - desired_relative_error) / maximum_singular_value
    solver_state.cumulative_kkt_passes += number_of_power_iterations
    
    # The opnorm is the correct one to use here, julia has a different implementation for the norm(). 
    # OPNORM NOT IMPLEMENTED FOR SPARSE IN JULIA, therefore we use maximum singular value instead! 
    # println("Calculated ||K|| using norm: ", opnorm(problem.constraint_matrix, 2)) 
  end

  if !(dwifob_params isa Nothing)
    (dwifob_solver_state, dwifob_matrix_cache) = initialize_dwifob_state(dwifob_params, primal_size, dual_size, maximum_singular_value)
  end  
        
  # Idealized number of KKT passes each time the termination criteria and
  # restart scheme is run. One of these comes from evaluating the gradient at
  # the average solution and evaluating the gradient at the current solution.
  # In practice this number is four.
  KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

  if params.scale_invariant_initial_primal_weight
    solver_state.primal_weight = select_initial_primal_weight(
      problem,
      ones(primal_size),
      ones(dual_size),
      params.primal_importance,
      params.verbosity,
    )
  else
    solver_state.primal_weight = params.primal_importance
  end

  primal_weight_update_smoothing =
    params.restart_params.primal_weight_update_smoothing

  iteration_stats = IterationStats[]
  
  # Debugging the algorithm
  println("The following should be true for M to be strictly positive: ")
  println(solver_state.step_size * solver_state.step_size * maximum_singular_value^2, " < ", 1)
  println("DWIFOB OPTION: ", params.dwifob_option)

  # Saving the initial step size and primal weight as global tau and sigma. # VERSION 1 HERE: 
  global tau_global = solver_state.step_size / solver_state.primal_weight
  global sigma_global = solver_state.step_size * solver_state.primal_weight
  println("tau_global: ", tau_global, " sigma_global: ", sigma_global)

  println("Initial primal weight: ", solver_state.primal_weight)

  start_time = time()
  # Basic algorithm refers to the primal and dual steps, and excludes restart
  # schemes and termination evaluation.
  time_spent_doing_basic_algorithm = 0.0

  # This variable is used in the adaptive restart scheme.
  last_restart_info = create_last_restart_info(
    problem,
    solver_state.current_primal_solution,
    solver_state.current_dual_solution,
  )

  # For termination criteria:
  termination_criteria = params.termination_criteria
  iteration_limit = termination_criteria.iteration_limit
  termination_evaluation_frequency = params.termination_evaluation_frequency

  # This flag represents whether a numerical error occurred during the algorithm
  # if it is set to true it will trigger the algorithm to terminate.
  solver_state.numerical_error = false
  display_iteration_stats_heading(params.verbosity)

  # For plotting: 
  iterate_plot_info = Vector{Float64}()
  rel_duality_gap_plot_info = Vector{Float64}()
  l2_primal_residual_plot_info = Vector{Float64}()
  l2_dual_residual_plot_info = Vector{Float64}()
  
  primal_iterates_plot_info = Vector{Vector{Float64}}()
  dual_iterates_plot_info = Vector{Vector{Float64}}()
  primal_averages_plot_info = Vector{Vector{Float64}}()
  dual_averages_plot_info = Vector{Vector{Float64}}()

  primal_hat_iterates_plot_info = Vector{Vector{Float64}}()
  dual_hat_iterates_plot_info = Vector{Vector{Float64}}()
  primal_deviation_plot_info = Vector{Vector{Float64}}()
  dual_deviation_plot_info = Vector{Vector{Float64}}()

  iteration = 0
  
  println("Dwifob Restart scheme: ", params.dwifob_restart, ", restart frequency: ", params.dwifob_restart_frequency)
  while true
    iteration += 1

    # Evaluate the iteration stats at frequency
    # termination_evaluation_frequency, when the iteration_limit is reached,
    # or if a numerical error occurs at the previous iteration.
    if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
       iteration == iteration_limit + 1 ||
       iteration <= 10 ||
       solver_state.numerical_error
      # TODO: Experiment with evaluating every power of two iterations.
      # This ensures that we do sufficient primal weight updates in the initial
      # stages of the algorithm.
      solver_state.cumulative_kkt_passes +=
        KKT_PASSES_PER_TERMINATION_EVALUATION
      # Compute the average solution since the last restart point.
      if solver_state.numerical_error ||
         solver_state.solution_weighted_avg.sum_primal_solutions_count == 0 ||
         solver_state.solution_weighted_avg.sum_dual_solutions_count == 0
        avg_primal_solution = solver_state.current_primal_solution
        avg_dual_solution = solver_state.current_dual_solution
      else
        avg_primal_solution, avg_dual_solution =
          compute_average(solver_state.solution_weighted_avg)
      end

      current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        params.termination_criteria,
        params.record_iteration_stats,
        avg_primal_solution,
        avg_dual_solution,
        iteration,
        time() - start_time,
        solver_state.cumulative_kkt_passes,
        termination_criteria.eps_optimal_absolute,
        termination_criteria.eps_optimal_relative,
        solver_state.step_size,
        solver_state.primal_weight,
        POINT_TYPE_AVERAGE_ITERATE,
      )
      
      if (params.should_save_convergence_data)       
        push!(iterate_plot_info, iteration)
        push!(rel_duality_gap_plot_info, current_iteration_stats.convergence_information[1].relative_optimality_gap)        
        push!(l2_primal_residual_plot_info, current_iteration_stats.convergence_information[1].l2_primal_residual)
        push!(l2_dual_residual_plot_info, current_iteration_stats.convergence_information[1].l2_dual_residual)
        
        if (params.should_save_detailed_data)
          push!(primal_iterates_plot_info, solver_state.current_primal_solution)
          push!(dual_iterates_plot_info, solver_state.current_dual_solution)
          push!(primal_averages_plot_info, avg_primal_solution)
          push!(dual_averages_plot_info, avg_dual_solution)

          # The dwifob specific parameters:
          if (isempty(dwifob_solver_state.x_hat_iterates))
            push!(primal_hat_iterates_plot_info, solver_state.current_primal_solution)
            push!(dual_hat_iterates_plot_info, solver_state.current_dual_solution)
          else 
            push!(primal_hat_iterates_plot_info, last(dwifob_solver_state.x_hat_iterates))
            push!(dual_hat_iterates_plot_info, last(dwifob_solver_state.y_hat_iterates))
          end 
          push!(primal_deviation_plot_info, dwifob_solver_state.current_primal_deviation)
          push!(dual_deviation_plot_info, dwifob_solver_state.current_dual_deviation)          
        end
      end 

      method_specific_stats = current_iteration_stats.method_specific_stats
      method_specific_stats["time_spent_doing_basic_algorithm"] =
        time_spent_doing_basic_algorithm

      primal_norm_params, dual_norm_params = define_norms(
        primal_size,
        dual_size,
        solver_state.step_size,
        solver_state.primal_weight,
      )
      update_objective_bound_estimates(
        current_iteration_stats.method_specific_stats,
        problem,
        avg_primal_solution,
        avg_dual_solution,
        primal_norm_params,
        dual_norm_params,
      )
      # Check the termination criteria.
      termination_reason = check_termination_criteria(
        termination_criteria,
        qp_cache,
        current_iteration_stats,
      )
      if solver_state.numerical_error && termination_reason == false
        termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
      end

      # If we're terminating, record the iteration stats to provide final
      # solution stats.
      if params.record_iteration_stats || termination_reason != false
        push!(iteration_stats, current_iteration_stats)
      end

      # Print table.
      if print_to_screen_this_iteration(
        termination_reason,
        iteration,
        params.verbosity,
        termination_evaluation_frequency,
      )
        display_iteration_stats(current_iteration_stats, params.verbosity)
      end

      if termination_reason != false
        # ** Terminate the algorithm **
        # This is the only place the algorithm can terminate. Please keep it
        # this way.
        pdhg_final_log(
          problem,
          avg_primal_solution,
          avg_dual_solution,
          params.verbosity,
          iteration,
          termination_reason,
          current_iteration_stats,
        )
        
        if (params.should_save_convergence_data)
          json_output_file = chop(output_file, head = 10, tail = 0) # Removes the "./results/" start of the string.  
          json_output_file_complete = "./results/datapoints/$(json_output_file).json"
          println("saving convergence information to: ", json_output_file_complete)
          
          plot_dict = Dict()
          plot_dict["iterations"] = iterate_plot_info
          plot_dict["rel_duality_gap"] = rel_duality_gap_plot_info
          plot_dict["l2_primal_residual"] = l2_primal_residual_plot_info
          plot_dict["l2_dual_residual"] = l2_dual_residual_plot_info
          
          # Write convergence results to file: 
          open(json_output_file_complete, "w") do f
            JSON3.pretty(f, plot_dict) 
          end
        
          if (params.should_save_detailed_data)
            json_detailed_output_file_complete = "./results/datapoints/$(json_output_file)_detailed.json"
            println("saving detailed convergence information to:\n", 
                    json_detailed_output_file_complete)

            plot_dict_detailed = Dict()
            plot_dict_detailed["primal_iterates"] = primal_iterates_plot_info
            plot_dict_detailed["dual_iterates"] = dual_iterates_plot_info
            plot_dict_detailed["primal_averages"] = primal_averages_plot_info
            plot_dict_detailed["dual_averages"] = dual_averages_plot_info

            # The dwifob info: 
            plot_dict_detailed["primal_hat_iterates"] = primal_hat_iterates_plot_info
            plot_dict_detailed["dual_hat_iterates"] = dual_hat_iterates_plot_info
            plot_dict_detailed["primal_deviations"] = primal_deviation_plot_info
            plot_dict_detailed["dual_deviations"] = dual_deviation_plot_info

            # Write detailed results to file: 
            open(json_detailed_output_file_complete, "w") do f
              JSON3.pretty(f, plot_dict_detailed) 
            end
          end
        end

        if (params.save_solution_json)
          json_output_file = chop(output_file, head = 10, tail = 0) # Removes the "./results/" start of the string.  
          json_solution_output_file_complete = "./results/datapoints/$(json_output_file)_solution.json"
          println("saving last iterate to: ", json_solution_output_file_complete)
          
          # Uncomment these lines when we want to store an accurate solution only. # FIXME: Make this a setting instead. 
          solution_dict = Dict()
          solution_dict["primal_solution"] = solver_state.current_primal_solution
          solution_dict["dual_solution"] = solver_state.current_dual_solution

          # Write detailed results to file: 
          open(json_solution_output_file_complete, "w") do f
            JSON3.pretty(f, solution_dict) 
          end
        end

        return unscaled_saddle_point_output(
          scaled_problem,
          avg_primal_solution,
          avg_dual_solution,
          termination_reason,
          iteration - 1,
          iteration_stats,
        )
      end

      current_iteration_stats.restart_used = run_restart_scheme(
        problem,
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        last_restart_info,
        iteration - 1,
        primal_norm_params,
        dual_norm_params,
        solver_state.primal_weight,
        params.verbosity,
        params.restart_params,
      )
      # println("Restart got: ", current_iteration_stats.restart_used)
      if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
        solver_state.primal_weight = compute_new_primal_weight(
          last_restart_info,
          solver_state.primal_weight,
          primal_weight_update_smoothing,
          params.verbosity,
        )
        solver_state.ratio_step_sizes = 1.0
        if (params.dwifob_restart == "PDLP")
          # println("Restarted dwifob")
          dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
        end
      end
      if current_iteration_stats.restart_used == RESTART_CHOICE_RESTART_TO_AVERAGE
        solver_state.current_dual_product =
        problem.constraint_matrix' * solver_state.current_dual_solution
        # println("Restarted to average.")
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
      end
    end

    #### The restart section for constant restarts of dwifob
    if (iteration > 10 && mod(iteration - 1, params.dwifob_restart_frequency) == 0 )
      if (params.dwifob_restart == "constant")
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
      elseif (params.dwifob_restart == "NOFOB")
        new_x = dwifob_solver_state.current_u_hat_x_deviation_sum
        new_y = dwifob_solver_state.current_u_hat_y_deviation_sum
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
        if (params.dwifob_option == "alt_C")
          solver_state.current_primal_solution = -new_x
          solver_state.current_dual_solution = -new_y  
        else 
          solver_state.current_primal_solution = new_x
          solver_state.current_dual_solution = new_y
        end
      end
    end
    time_spent_doing_basic_algorithm_checkpoint = time()

    if params.verbosity >= 6 && print_to_screen_this_iteration(
      false, # termination_reason
      iteration,
      params.verbosity,
      termination_evaluation_frequency,
    )
      pdhg_specific_log(
        problem,
        iteration,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.required_ratio,
        solver_state.primal_weight,
      )
    end
    
    if params.steering_vectors
      if params.dwifob_option == "alt_A"
        take_dwifob_step_alt_A(params.step_size_policy_params, problem, solver_state, dwifob_solver_state)
      elseif params.dwifob_option == "alt_B"
        take_dwifob_step_alt_B(params.step_size_policy_params, problem, solver_state, dwifob_solver_state)
      elseif params.dwifob_option == "alt_C"
        take_dwifob_step_alt_C(params.step_size_policy_params, problem, solver_state, dwifob_solver_state)
      elseif params.fast_dwifob
        take_dwifob_step_efficient(params.step_size_policy_params, problem, solver_state, dwifob_solver_state, dwifob_matrix_cache)
      else
        take_dwifob_step(params.step_size_policy_params, problem, solver_state, dwifob_solver_state) 
      end      
    else 
      take_step(params.step_size_policy_params, problem, solver_state)
    end

    time_spent_doing_basic_algorithm +=
      time() - time_spent_doing_basic_algorithm_checkpoint
  end
end

# TODO: Solve this using efficient QR factorization, will not affect results but only speed.
"""
  Function returning approximate solution to the Andersson acceleration step in the DWIFOB algorithm.
  Uses implicit regularization to handle singular R_k^T R_k matrices.
"""
function calculate_anderson_acceleration(
  solver_state::PdhgSolverState,
  R_k::Matrix{Float64},
  m_k::Int64,
)
  if m_k == 0
    return [1.0]
  else      
    # Since KKT passes are used as a measure of how large calculations we do in the algorithm, 
    # we need to modify this in the AA step as well. 
    # One KKT pass is equal to one matrix multiplication of the M matrix, 
    # or one with K^T and one with K. That would be the cost of 
    # This is equivalent to multiplying with a (n = primal_size + dual_size)
    # 1 KKT pass = p * d * d + p * d * p = (d+p) (p * d)
    # Calculating the relative cost of the AA step: 
    p = size(solver_state.current_primal_solution)[1]
    d = size(solver_state.current_primal_solution)[1]
    n = p + d
    cost_approx = n*m_k + m_k^2 # (See Anderson Acceleration of Proximal Gradient Methods)
    cost_relative_KKT = cost_approx / (n * p * d)
    # We can look at some results and then decide if above reflects the complexity of the AA step. 
    # (It shows the theoretical one with better implementation, which is extremely close to 0)
    solver_state.cumulative_kkt_passes += cost_relative_KKT
    ones_corr_dim = ones(size(R_k)[2], 1) 
    
    x = (R_k' * R_k + 1e-4*1.0I)\ones_corr_dim  
    alpha = x / (ones_corr_dim' * x)
    # println("m_k: ", m_k, " dim alpha: ", size(alpha))
    return alpha
  end
end

"""
  Function returning approximate solution to the Andersson acceleration step in the DWIFOB algorithm.
  Uses iterative refinement to calculate it to greater accuracy.
"""
function calculate_anderson_acceleration_prox_iterations(
  solver_state::PdhgSolverState,
  R_k::Matrix{Float64},
  m_k::Int64,
  iterations::Int64,
)
  if m_k == 0
    return [1.0]
  else      
    ones_corr_dim = ones(size(R_k)[2], 1) 
    x_i = (R_k' * R_k + 1e-8*1.0I)\ones_corr_dim  
    for i in 2:iterations
      x_i = x_i + (R_k' * R_k + 1e-8*1.0I)\(ones_corr_dim - R_k' * R_k * x_i)
    end
    alpha = x_i / (ones_corr_dim' * x_i)
    return alpha
  end
end

function squared_norm_M(
  x::Vector{Float64},
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
)
  # Forming the M matrix, we do this for the current step sizes used (depending on the primal weight)
  tau = solver_state.step_size / solver_state.primal_weight
  sigma = solver_state.step_size * solver_state.primal_weight

  M = [1.0I tau*problem.constraint_matrix; tau*problem.constraint_matrix' 1.0I*tau/sigma] 

  # Calculating the norm with respect to the calculated M matrix:
  return x' * M * x
end

""" 
Calculates the norm: (|| [x, y] ||_M)^2 
with respect to the M matrix efficiently using cached values of K*x 
"""
function squared_norm_M_fast(
  x::Vector{Float64},
  y::Vector{Float64},
  K_x::Vector{Float64},
  solver_state::PdhgSolverState,
)
  # ORIGINAL VERSION: 
  tau = solver_state.step_size / solver_state.primal_weight
  sigma = solver_state.step_size * solver_state.primal_weight
  return norm(x, 2)^2 + (tau/sigma) * norm(y, 2)^2 + 2 * tau * y' * K_x
  # GLOBTAUSIGMA VERSION: 
  # global tau_global
  # global sigma_global
  # return norm(x, 2)^2 + (tau_global/sigma_global) * norm(y, 2)^2 + 2 * tau_global * y' * K_x
  # 2-NORM VERSION: 
  # return norm(x, 2)^2 + norm(y, 2)^2
end 

"""Initializes the state for the structs required in dwifob."""
function initialize_dwifob_state(
  dwifob_params::DwifobParameters,
  primal_size::Int64,
  dual_size::Int64,
  maximum_singular_value::Float64,
)
  # Initializing DWIFOB solver struct:
  x_list = Vector{Vector{Float64}}()
  y_list = Vector{Vector{Float64}}()
  x_hat_list = Vector{Vector{Float64}}()
  y_hat_list = Vector{Vector{Float64}}()

  dwifob_solver_state = DwifobSolverState(
    dwifob_params.max_memory, # max_memory
    0,                        # current_iteration
    1,                        # lambda_k
    1,                        # lambda_next
    0.99,                     # zeta_k
    1e-4,                     # epsilon
    maximum_singular_value,   # maximum singular value of K.
    x_list,                   # primal_iterates
    y_list,                   # dual_iterates
    x_hat_list,               # primal_hat_iterates
    y_hat_list,               # dual_hat_iterates
    zeros(primal_size),       # current_primal_deviation
    zeros(dual_size),         # current_dual_deviation
    zeros(primal_size),       # current u_hat_x_deviation_sum
    zeros(dual_size),         # current u_hat_y_deviation_sum
  )

  # Initializing the matrix cache:
  K_x_list = Vector{Vector{Float64}}()
  KT_y_list = Vector{Vector{Float64}}()

  dwifob_matrix_cache = DwifobMatrixCache(
    K_x_list,                 # list of cached values of K_x 
    KT_y_list,                # list of cached valued of K^T y
    [0],                      # cache of K x 
    [0],                      # cache of K^T y 
    [0],                      # cache of K x_hat 
    [0],                      # cache of K^T y_hat 
    [0],                      # cache of K u_x 
  )
  return dwifob_solver_state, dwifob_matrix_cache
end

### DWIFOB Steps: ####
"""
Takes a step with constant step size using steering vectors.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.
"""
function take_dwifob_step(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  debugging=false
)
  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)
  K_trans_y_next = problem.constraint_matrix' * y_next

  # Update the solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next,
  )

  # Preparing the input for the Regularized Andersson Acceleration:
  push!(dwifob_solver_state.primal_iterates, x_next)
  push!(dwifob_solver_state.dual_iterates, y_next)
  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
  end 
  u_hat_x_next = x_next - u_hat_x_deviation_sum
  u_hat_y_next = y_next - u_hat_y_deviation_sum


  # Calculating the l^2_k factor: 
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  norm_argument = [p_x_k; p_y_k] - [x_k; y_k] + (2 * lambda_k - 2) / (4 - 2 * lambda_k) * [u_x_k; u_y_k]
  l_squared_k = multiplicative_factor * squared_norm_M(norm_argument, problem, solver_state)


  # Calculating the deviations for the next iteration:
  u_next_hat = [u_hat_x_next; u_hat_y_next]
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) 
  scaling_factor = scaling_factor / (dwifob_solver_state.epsilon + sqrt(squared_norm_M(u_next_hat, problem, solver_state)))

  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next

  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  solver_state.cumulative_kkt_passes += 3

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)
  dwifob_solver_state.current_u_hat_x_deviation_sum = u_hat_x_deviation_sum
  dwifob_solver_state.current_u_hat_y_deviation_sum = u_hat_y_deviation_sum
  
  if debugging
    println("#######################################")
    println("### At iteration: ", dwifob_solver_state.current_iteration, " we get the following: ###")
    println("x_hat_k: ", x_hat_k)
    println("u_x_next: ", u_x_next)
    println("primal gradient: ", primal_gradient)
    println("u_y_next: ", u_y_next)
    println("dual gradient: ", dual_gradient)
    println("")
    
    println("Scaling factor: ", scaling_factor)
    println("alpha: ", alpha_k)
    println("u_hat_x_next: ", u_hat_x_next)
    println("u_hat_y_next: ", u_hat_y_next)      
    println("l_squared: ", l_squared_k)
    println("")

    println("x_iterates: ", dwifob_solver_state.primal_iterates)

    K = problem.constraint_matrix
    print("K_x_iterates: [")
    for iterate in dwifob_solver_state.primal_iterates
      print(K * iterate, ", ")
    end
    print("]\n")
    println("weighted_K_x_sum: ", K * u_hat_x_deviation_sum)
    println("")

    println("K_u_hat_x_next: ", K * u_hat_x_next)
    println("K_u_x_next: ", K * u_x_next)
    println("K_x_next: ", K * x_next)
    println("K_x_hat_next: ", K * x_hat_next)

    println("#######################################")
  end
end

# TODO: Make this have its own code instead of relying on the functions to manipulate the dwifob state.
"""
Takes a step using the adaptive step size and steering vectors.
It modifies the third and fourth arguments: solver_state and dwifob_solver_state.
"""
function take_dwifob_step(
  step_params::AdaptiveStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  debugging=false
)
  step_size = solver_state.step_size
  done = false
  iter = 0

  while !done
    iter += 1
    solver_state.total_number_iterations += 1

    # Try to take DWIFOB Step here, but do not manipulate the solver states yet, 
    # we need the next primal, dual and dual product.: 
    (next_primal, next_dual, next_dual_product, p_x_k, p_y_k
      ) = get_next_dwifob_candidate(problem, solver_state, dwifob_solver_state)

    interaction, movement = compute_interaction_and_movement(
      solver_state,
      problem,
      next_primal,
      next_dual,
      next_dual_product,
    )
    # This one is incorrect but displays the lowest possible if using fast implementation. 
    solver_state.cumulative_kkt_passes += 1

    if movement == 0.0
      # The algorithm will terminate at the beginning of the next iteration
      solver_state.numerical_error = true
      break
    end
    # The proof of Theorem 1 requires movement / step_size >= interaction.
    if interaction > 0
      step_size_limit = movement / interaction
    else
      step_size_limit = Inf
    end

    if step_size <= step_size_limit
      update_dwifob_state(
        problem, 
        solver_state,
        dwifob_solver_state,
        next_primal,
        next_dual,
        next_dual_product,
        p_x_k,
        p_y_k,
      )
      done = true
    # If the first step that we took was too long, we need to uninitialize the dwifob solver state.
    elseif (dwifob_solver_state.current_iteration == 0) 
      popfirst!(dwifob_solver_state.x_hat_iterates)
      popfirst!(dwifob_solver_state.y_hat_iterates)
    end

    first_term = (step_size_limit * 
      (1 - (solver_state.total_number_iterations + 1)^(-step_params.reduction_exponent)))
    second_term = (step_size * 
      (1 + (solver_state.total_number_iterations + 1)^(-step_params.growth_exponent)))
    step_size = min(first_term, second_term, 1/(dwifob_solver_state.maximum_singular_value^2))
  end
  solver_state.step_size = step_size
 
  # println("The following should be true for M to be strictly positive: ")
  # println(step_size^2 * maximum_singular_value^2, " < ", 1)

end

"""
Takes a step with constant step size using steering vectors.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.
This function uses the fields for previously calculated values of K x and K^T y 
  in the DwifobSolverState 
"""
function take_dwifob_step_efficient(  
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  dwifob_matrix_cache::DwifobMatrixCache,
  debugging=false,
)
  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)

  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    # Initializing the regular dwifob lists:
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
    
    # Initializing the cached matrix products:
    K_x = problem.constraint_matrix * solver_state.current_primal_solution
    K_trans_y = problem.constraint_matrix' * solver_state.current_dual_solution
    # Keeping track of the KKT passes: 
    solver_state.cumulative_kkt_passes += 1
    
    dwifob_matrix_cache.K_x_current = K_x
    dwifob_matrix_cache.K_trans_y_current = K_trans_y
    dwifob_matrix_cache.K_x_hat_current = K_x
    dwifob_matrix_cache.K_trans_y_hat_current = K_trans_y
    
    K_u_x = 0 .* K_x      
    # FIXME: HACK This above is ugly and we want to do it better, with zeros instead. 
    # Idea below but does not work yet.
    # K_u_x = zeros(Float64, size(K_x, 1), 1) 
    
    # println("K_x dimensions: ", size(K_x, 1))
    # println("K_u_k_cur dimensions: ", size(K_u_x))
    dwifob_matrix_cache.K_u_x_current = K_u_x
  end

  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  K_x_k = dwifob_matrix_cache.K_x_current
  K_trans_y_k = dwifob_matrix_cache.K_trans_y_current

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - dwifob_matrix_cache.K_trans_y_hat_current
  tau = solver_state.step_size / solver_state.primal_weight
  p_x_k = x_hat_k - tau * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating new K matrix product: 
  K_p_x_k = problem.constraint_matrix * p_x_k

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - (2 * K_p_x_k) + dwifob_matrix_cache.K_x_hat_current  
  sigma = solver_state.step_size * solver_state.primal_weight
  p_y_k = y_hat_k + sigma * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating new K matrix product: 
  K_trans_p_y_k = problem.constraint_matrix' * p_y_k

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)
  # Calculating: K x_next and K^T y_next 
  K_x_next = K_x_k + lambda_k * (K_p_x_k - dwifob_matrix_cache.K_x_hat_current)
  K_trans_y_next = K_trans_y_k + lambda_k * (K_trans_p_y_k - dwifob_matrix_cache.K_trans_y_hat_current)

  # Update the solver states: 
  # The regular solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next, # We already have this in the dwifob struct but we need it here as well to track averages for the PDLP restarts.
  )

  # The dwifob specific states: 
  # Keeping track of the KKT passes: 
  solver_state.cumulative_kkt_passes += 1
  # The primal and dual solutions: 
  solver_state.current_primal_solution = x_next
  solver_state.current_dual_solution = y_next  
  push!(dwifob_solver_state.primal_iterates, x_next)
  push!(dwifob_solver_state.dual_iterates, y_next)
  # The cached matrix products: 
  push!(dwifob_matrix_cache.K_x_iterates, K_x_next)
  push!(dwifob_matrix_cache.K_trans_y_iterates, K_trans_y_next)
  dwifob_matrix_cache.K_x_current = K_x_next
  dwifob_matrix_cache.K_trans_y_current = K_trans_y_next

  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)

    # Handling the cached matrix products:
    popfirst!(dwifob_matrix_cache.K_x_iterates)
    popfirst!(dwifob_matrix_cache.K_trans_y_iterates)
  end

  # Preparing the input for the Regularized Andersson Acceleration:
  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  # And weighted sums of cached matrix products: 
  weighted_K_x_sum = zeros(size(K_x_k))
  weighted_K_trans_y_sum = zeros(size(K_trans_y_k))

  for (primal_i, dual_i, alpha_i, K_x_i, K_trans_y_i) in zip(
    dwifob_solver_state.primal_iterates, 
    dwifob_solver_state.dual_iterates, 
    alpha_k, 
    dwifob_matrix_cache.K_x_iterates, 
    dwifob_matrix_cache.K_trans_y_iterates
  )
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
    weighted_K_x_sum += K_x_i * alpha_i
    weighted_K_trans_y_sum += K_trans_y_i * alpha_i
  end 
  u_hat_x_next = x_next - u_hat_x_deviation_sum
  u_hat_y_next = y_next - u_hat_y_deviation_sum

  # Calculating the matrix products without matrix multiplication: 
  K_u_hat_x_next = K_x_next - weighted_K_x_sum
  K_u_hat_y_next = K_trans_y_next - weighted_K_trans_y_sum

  # Calculating the l^2_k factor: 
  norm_factor = (2 * lambda_k - 2) / (4 - 2 * lambda_k)
  norm_argument_x = p_x_k - x_k + norm_factor * u_x_k
  norm_argument_y = p_y_k - y_k + norm_factor * u_y_k

  norm_argument_K_x = K_p_x_k - K_x_k + norm_factor * dwifob_matrix_cache.K_u_x_current
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  l_squared_k = multiplicative_factor * squared_norm_M_fast(norm_argument_x, norm_argument_y, norm_argument_K_x, solver_state)

  # HACK: If l_squared is very close to 0 but negative, we round it to 0. 
  # For some reason we had to add this when restarting with frequency=80, this did not help...
  # if (l_squared_k < 0 && l_squared_k > -0.01)
  #   l_squared_k = 0
  # end
  sqrt_arg = squared_norm_M_fast(u_hat_x_next, u_hat_y_next, K_u_hat_x_next, solver_state)
  
  # if (sqrt_arg < 0 && sqrt_arg > -0.01)
  #   sqrt_arg = 0
  # end

  # Calculating the deviations for the next iteration:
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) / (
    dwifob_solver_state.epsilon + 
    sqrt(sqrt_arg)
  )

  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next
  
  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1

  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)

  # Calculating the next chached matrix products: 
  # The deviation vectors: 
  K_u_x_next = scaling_factor * K_u_hat_x_next 
  K_trans_u_y_next = scaling_factor * K_u_hat_y_next  
  # The hat vectors: 
  K_x_hat_next = K_x_next + K_u_x_next 
  K_trans_y_hat_next = K_trans_y_next + K_trans_u_y_next

  # The past deviation sums, these are used in NOFOB type restarts. 
  dwifob_solver_state.current_u_hat_x_deviation_sum = u_hat_x_deviation_sum
  dwifob_solver_state.current_u_hat_y_deviation_sum = u_hat_y_deviation_sum

  # Storing the cached matrix products in the struct:
  dwifob_matrix_cache.K_u_x_current = K_u_x_next
  dwifob_matrix_cache.K_x_hat_current = K_x_hat_next
  dwifob_matrix_cache.K_trans_y_hat_current = K_trans_y_hat_next

  if debugging
    println("#######################################")
    println("### At iteration: ", dwifob_solver_state.current_iteration, " we get the following: ###")
    println("x_hat_k: ", x_hat_k)
    println("u_x_next: ", u_x_next)
    println("primal gradient: ", primal_gradient)
    println("u_y_next: ", u_y_next)
    println("dual gradient: ", dual_gradient)
    println("")
    
    println("Scaling factor: ", scaling_factor)
    println("alpha: ", alpha_k)
    println("u_hat_x_next: ", u_hat_x_next)
    println("u_hat_y_next: ", u_hat_y_next)      
    println("l_squared: ", l_squared_k)
    println("")

    println("x_iterates: ", dwifob_solver_state.primal_iterates)

    println("K_x_iterates: ", dwifob_matrix_cache.K_x_iterates)
    println("weighted_K_x_sum: ", weighted_K_x_sum)
    println("")

    println("K_u_hat_x_next: ", K_u_hat_x_next)
    println("K_u_x_next: ", K_u_x_next)
    println("K_x_next: ", K_x_next)
    println("K_x_hat_next: ", K_x_hat_next)

    println("#######################################")
  end
end

"""
Takes a step using the adaptive step size and steering vectors.
It modifies the third and fourth arguments: solver_state and dwifob_solver_state.
"""
function take_dwifob_step_efficient(
  step_params::AdaptiveStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  dwifob_matrix_cache::DwifobMatrixCache,
  debugging=false
)
  step_size = solver_state.step_size
  done = false
  iter = 0

  while !done
    iter += 1
    solver_state.total_number_iterations += 1

    m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)

    # Initializing the hat variables of the algorithm:
    if (dwifob_solver_state.current_iteration == 0)
      # Initializing the regular dwifob lists:
      push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
      push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
      
      # Initializing the cached matrix products:
      K_x = problem.constraint_matrix * solver_state.current_primal_solution
      K_trans_y = problem.constraint_matrix' * solver_state.current_dual_solution
      # Keeping track of the KKT passes: 
      solver_state.cumulative_kkt_passes += 1
      
      dwifob_matrix_cache.K_x_current = K_x
      dwifob_matrix_cache.K_trans_y_current = K_trans_y
      dwifob_matrix_cache.K_x_hat_current = K_x
      dwifob_matrix_cache.K_trans_y_hat_current = K_trans_y
      
      K_u_x = 0 .* K_x      
      # FIXME: HACK This above is ugly and we want to do it better, with zeros instead. 
      # Idea below but does not work yet.
      # K_u_x = zeros(Float64, size(K_x, 1), 1) 
      
      # println("K_x dimensions: ", size(K_x, 1))
      # println("K_u_k_cur dimensions: ", size(K_u_x))
      dwifob_matrix_cache.K_u_x_current = K_u_x
    end
  
    # Extracting some variables from the solver state struct
    # for clearer and more concise code:  
    x_hat_k = last(dwifob_solver_state.x_hat_iterates)
    y_hat_k = last(dwifob_solver_state.y_hat_iterates)
  
    x_k = solver_state.current_primal_solution
    y_k = solver_state.current_dual_solution
  
    K_x_k = dwifob_matrix_cache.K_x_current
    K_trans_y_k = dwifob_matrix_cache.K_trans_y_current
  
    lambda_k = dwifob_solver_state.lambda_k
  
    if isnan(x_hat_k[1]) 
      println("Got NaN in iterates, aborting...")
      exit(1)
    end
  
    # Calculating the primal "pseudogradient" (p_x_k) value:
    primal_gradient = problem.objective_vector - dwifob_matrix_cache.K_trans_y_hat_current
    tau = step_size / solver_state.primal_weight
    p_x_k = x_hat_k - tau * primal_gradient
    project_primal!(p_x_k, problem)
  
    # Calculating new K matrix product: 
    K_p_x_k = problem.constraint_matrix * p_x_k
  
    # Calculating the dual "pseudogradient" (p_y_k) value: 
    dual_gradient = problem.right_hand_side - (2 * K_p_x_k) + dwifob_matrix_cache.K_x_hat_current  
    sigma = step_size * solver_state.primal_weight
    p_y_k = y_hat_k + sigma * dual_gradient
    project_dual!(p_y_k, problem)
  
    # Calculating new K matrix product: 
    K_trans_p_y_k = problem.constraint_matrix' * p_y_k
  
    # Calculating the next iterates:
    x_next = x_k + lambda_k * (p_x_k - x_hat_k)
    y_next = y_k + lambda_k * (p_y_k - y_hat_k)
    # Calculating: K^T y_next 
    K_trans_y_next = K_trans_y_k + lambda_k * (K_trans_p_y_k - dwifob_matrix_cache.K_trans_y_hat_current)  
    
    interaction, movement = compute_interaction_and_movement(
      solver_state,
      problem,
      x_next,
      y_next,
      K_trans_y_next,
    )

    # Keeping track of the KKT passes. 
    solver_state.cumulative_kkt_passes += 1

    # if movement == 0.0
    #   # The algorithm will terminate at the beginning of the next iteration
    #   solver_state.numerical_error = true
    #   break
    # end
    # The proof of Theorem 1 requires movement / step_size >= interaction.
    if interaction > 0
      step_size_limit = movement / interaction
    else
      step_size_limit = Inf
    end

    if step_size <= step_size_limit
      # Update the solver states: 
      # The regular solver state: 
      update_solution_in_solver_state(
        solver_state,
        x_next,
        y_next,
        K_trans_y_next, # We already have this in the dwifob struct but we need it here as well to track averages for the PDLP restarts.
      )
      K_x_next = K_x_k + lambda_k * (K_p_x_k - dwifob_matrix_cache.K_x_hat_current)

      # The dwifob specific states: 
      # The primal and dual solutions: 
      solver_state.current_primal_solution = x_next
      solver_state.current_dual_solution = y_next  
      push!(dwifob_solver_state.primal_iterates, x_next)
      push!(dwifob_solver_state.dual_iterates, y_next)
      # The cached matrix products: 
      push!(dwifob_matrix_cache.K_x_iterates, K_x_next)
      push!(dwifob_matrix_cache.K_trans_y_iterates, K_trans_y_next)
      dwifob_matrix_cache.K_x_current = K_x_next
      dwifob_matrix_cache.K_trans_y_current = K_trans_y_next
    
      if (m_k < dwifob_solver_state.current_iteration) 
        popfirst!(dwifob_solver_state.primal_iterates)
        popfirst!(dwifob_solver_state.dual_iterates)
        popfirst!(dwifob_solver_state.x_hat_iterates)
        popfirst!(dwifob_solver_state.y_hat_iterates)
    
        # Handling the cached matrix products:
        popfirst!(dwifob_matrix_cache.K_x_iterates)
        popfirst!(dwifob_matrix_cache.K_trans_y_iterates)
      end
    
      # Preparing the input for the Regularized Andersson Acceleration:
      # Calculating R_k (linear combination of inertial terms) for the RAA:   
      R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
      R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
      # Converting to matrixes for easier calculations: 
      R_k_primal = reduce(hcat, R_k_primal)
      R_k_dual = reduce(hcat, R_k_dual)
      # Finally, forming R_k and calculating the RAA:
      R_k = vcat(R_k_primal, R_k_dual)
      alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)
    
      # Calculating the hat_deviations: 
      u_hat_x_deviation_sum = zeros(size(x_k))
      u_hat_y_deviation_sum = zeros(size(y_k))
      # And weighted sums of cached matrix products: 
      weighted_K_x_sum = zeros(size(K_x_k))
      weighted_K_trans_y_sum = zeros(size(K_trans_y_k))
      for (primal_i, dual_i, alpha_i, K_x_i, K_trans_y_i) in zip(
        dwifob_solver_state.primal_iterates, 
        dwifob_solver_state.dual_iterates, 
        alpha_k, 
        dwifob_matrix_cache.K_x_iterates, 
        dwifob_matrix_cache.K_trans_y_iterates
      )
        u_hat_x_deviation_sum += primal_i * alpha_i
        u_hat_y_deviation_sum += dual_i * alpha_i
        weighted_K_x_sum += K_x_i * alpha_i
        weighted_K_trans_y_sum += K_trans_y_i * alpha_i
      end 
      u_hat_x_next = x_next - u_hat_x_deviation_sum
      u_hat_y_next = y_next - u_hat_y_deviation_sum
    
      # Calculating the matrix products without matrix multiplication: 
      K_u_hat_x_next = K_x_next - weighted_K_x_sum
      K_u_hat_y_next = K_trans_y_next - weighted_K_trans_y_sum
    
      # Calculating the l^2_k factor: 
      lambda_next = dwifob_solver_state.lambda_next
      norm_factor = (2 * lambda_k - 2) / (4 - 2 * lambda_k)
      norm_argument_x = p_x_k - x_k + norm_factor * dwifob_solver_state.current_primal_deviation
      norm_argument_y = p_y_k - y_k + norm_factor * dwifob_solver_state.current_dual_deviation
    
      norm_argument_K_x = K_p_x_k - K_x_k + norm_factor * dwifob_matrix_cache.K_u_x_current
      multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
      l_squared_k = multiplicative_factor * squared_norm_M_fast(norm_argument_x, norm_argument_y, norm_argument_K_x, solver_state)

      sqrt_arg = squared_norm_M_fast(u_hat_x_next, u_hat_y_next, K_u_hat_x_next, solver_state)  
      # Calculating the deviations for the next iteration:
      scaling_factor = dwifob_solver_state.zeta_k * sqrt(max(0, l_squared_k)) / (
        dwifob_solver_state.epsilon + 
        sqrt(max(0, sqrt_arg))
      )
    
      u_x_next = scaling_factor * u_hat_x_next
      u_y_next = scaling_factor * u_hat_y_next
      
      # Calculating the hat iterates:
      x_hat_next = x_next + u_x_next
      y_hat_next = y_next + u_y_next
    
      # Updating the changes in the mutable dwifob struct before the next iteration:
      dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
    
      dwifob_solver_state.current_primal_deviation = u_x_next
      dwifob_solver_state.current_dual_deviation = u_y_next
      push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
      push!(dwifob_solver_state.y_hat_iterates, y_hat_next)
    
      # Calculating the next chached matrix products: 
      # The deviation vectors: 
      K_u_x_next = scaling_factor * K_u_hat_x_next 
      K_trans_u_y_next = scaling_factor * K_u_hat_y_next  
      # The hat vectors: 
      K_x_hat_next = K_x_next + K_u_x_next 
      K_trans_y_hat_next = K_trans_y_next + K_trans_u_y_next
    
      # The past deviation sums, these are used in NOFOB type restarts. 
      dwifob_solver_state.current_u_hat_x_deviation_sum = u_hat_x_deviation_sum
      dwifob_solver_state.current_u_hat_y_deviation_sum = u_hat_y_deviation_sum
      
      # Storing the cached matrix products in the struct:
      dwifob_matrix_cache.K_u_x_current = K_u_x_next
      dwifob_matrix_cache.K_x_hat_current = K_x_hat_next
      dwifob_matrix_cache.K_trans_y_hat_current = K_trans_y_hat_next
      done = true
    # If the first step that we took was too long, we need to uninitialize the dwifob solver state.
    elseif (dwifob_solver_state.current_iteration == 0) 
      popfirst!(dwifob_solver_state.x_hat_iterates)
      popfirst!(dwifob_solver_state.y_hat_iterates)
    end

    first_term = (step_size_limit * 
      (1 - (solver_state.total_number_iterations + 1)^(-step_params.reduction_exponent)))
    second_term = (step_size * 
      (1 + (solver_state.total_number_iterations + 1)^(-step_params.growth_exponent)))
    # third_term = 1/(dwifob_solver_state.maximum_singular_value^2 + 1e-6)
    step_size = min(first_term, second_term) #, third_term)
    if (mod(solver_state.total_number_iterations, 5) == 0)
      println("At iteration: ", solver_state.total_number_iterations, " accepted step of: ", step_size^2 * dwifob_solver_state.maximum_singular_value^2)
    end
  end
  solver_state.step_size = step_size
end

"""
  Calculates the next iterates using the DWIFOB algorithm. 
"""
function get_next_dwifob_candidate(
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
)
  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  lambda_k = dwifob_solver_state.lambda_k

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k) 
  K_trans_y_next = problem.constraint_matrix' * y_next

  return x_next, y_next, K_trans_y_next, p_x_k, p_y_k
end

"""
  Prepares the solver states for the next iteration with DWIFOB. 
"""
function update_dwifob_state(
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  x_next::Vector{Float64}, 
  y_next::Vector{Float64},
  K_trans_y_next::Vector{Float64},
  p_x_k:: Vector{Float64},
  p_y_k:: Vector{Float64},
)
  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  # Update the solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next, 
  )

  # Preparing the input for the Regularized Andersson Acceleration:
  push!(dwifob_solver_state.primal_iterates, x_next)
  push!(dwifob_solver_state.dual_iterates, y_next)
  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  ## FIXME: Here we have an error, the iterates are of different dimension, 
  # we have to debug this, and then when it works, we need to try removing the abs in the sqrt(l)
  #   (see the other FIXME below.)
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
  end 
  u_hat_x_next = x_next - u_hat_x_deviation_sum
  u_hat_y_next = y_next - u_hat_y_deviation_sum

  # Calculating the l^2_k factor: 
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  norm_argument = [p_x_k; p_y_k] - [x_k; y_k] + (2 * lambda_k - 2) / (4 - 2 * lambda_k) * [u_x_k; u_y_k]
  l_squared_k = multiplicative_factor * squared_norm_M(norm_argument, problem, solver_state)

  # Calculating the deviations for the next iteration:
  u_next_hat = [u_hat_x_next; u_hat_y_next]
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) # FIXME: This does not seem good, we will have to address this at some point...
  scaling_factor = scaling_factor / (dwifob_solver_state.epsilon + sqrt(squared_norm_M(u_next_hat, problem, solver_state)))

  # scaling_factor = min(1, scaling_factor)
  # println("Scaling factor: ", scaling_factor)
  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next

  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  # solver_state.cumulative_kkt_passes += 3

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)
end


"""
Takes a step with constant step size using steering vectors.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.

Different version of DWIFOB:
  - This one where p_x_k and p_y_k are inputted to the RAA instead of x_k+1 and y_k+1
  - For simplicity, we use the same names in the DWIFOB cache structs as the original implementation. 
    - primal/dual iterates store the primal/dual pseudogradients instead in this version.
"""
function take_dwifob_step_alt_A(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  debugging=false,
)
  # Initializing the hat variables of the algorithm:
  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)

  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)

  # Update the solver states: 
  solver_state.current_primal_solution = x_next
  solver_state.current_dual_solution = y_next

  # Preparing the input for the Regularized Andersson Acceleration:
  push!(dwifob_solver_state.primal_iterates, p_x_k)
  push!(dwifob_solver_state.dual_iterates, p_y_k)
  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
  end 
  u_hat_x_next = p_x_k - u_hat_x_deviation_sum
  u_hat_y_next = p_y_k - u_hat_y_deviation_sum

  # Calculating the l^2_k factor: 
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  norm_argument = [p_x_k; p_y_k] - [x_k; y_k] + (2 * lambda_k - 2) / (4 - 2 * lambda_k) * [u_x_k; u_y_k]
  l_squared_k = multiplicative_factor * squared_norm_M(norm_argument, problem, solver_state)

  # Calculating the deviations for the next iteration:
  u_next_hat = [u_hat_x_next; u_hat_y_next]
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) 
  scaling_factor = scaling_factor / (dwifob_solver_state.epsilon + sqrt(squared_norm_M(u_next_hat, problem, solver_state)))

  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next

  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  solver_state.cumulative_kkt_passes += 3

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)

  if debugging
    println("#######################################")
    println("### At iteration: ", dwifob_solver_state.current_iteration, " we get the following: ###")
    println("x_hat_k: ", x_hat_k)
    println("u_x_next: ", u_x_next)
    println("primal gradient: ", primal_gradient)
    println("u_y_next: ", u_y_next)
    println("dual gradient: ", dual_gradient)
    println("")
    
    println("Scaling factor: ", scaling_factor)
    println("alpha: ", alpha_k)
    println("u_hat_x_next: ", u_hat_x_next)
    println("u_hat_y_next: ", u_hat_y_next)      
    println("l_squared: ", l_squared_k)
    println("")

    println("x_iterates: ", dwifob_solver_state.primal_iterates)

    K = problem.constraint_matrix
    print("K_x_iterates: [")
    for iterate in dwifob_solver_state.primal_iterates
      print(K * iterate, ", ")
    end
    print("]\n")
    println("weighted_K_x_sum: ", K * u_hat_x_deviation_sum)
    println("")

    println("K_u_hat_x_next: ", K * u_hat_x_next)
    println("K_u_x_next: ", K * u_x_next)
    println("K_x_next: ", K * x_next)
    println("K_x_hat_next: ", K * x_hat_next)

    println("#######################################")
  end
end

"""
Takes a step with constant step size using steering vectors.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.

Different version of DWIFOB:
  - This one where we take inspiration from the NOFOB article. 
"""
function take_dwifob_step_alt_B(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  debugging=false
)
  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)

  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)

  # Update the solver states: 
  solver_state.current_primal_solution = x_next
  solver_state.current_dual_solution = y_next

  # Preparing the input for the Regularized Andersson Acceleration:
  push!(dwifob_solver_state.primal_iterates, x_next)
  push!(dwifob_solver_state.dual_iterates, y_next)
  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
  end 
  u_hat_x_next = -x_next + u_hat_x_deviation_sum
  u_hat_y_next = -y_next + u_hat_y_deviation_sum

  # Calculating the l^2_k factor: 
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  norm_argument = [p_x_k; p_y_k] - [x_k; y_k] + (2 * lambda_k - 2) / (4 - 2 * lambda_k) * [u_x_k; u_y_k]
  l_squared_k = multiplicative_factor * squared_norm_M(norm_argument, problem, solver_state)

  # Calculating the deviations for the next iteration:
  u_next_hat = [u_hat_x_next; u_hat_y_next]
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) 
  scaling_factor = scaling_factor / (dwifob_solver_state.epsilon + sqrt(squared_norm_M(u_next_hat, problem, solver_state)))

  # This line is new compared to originial implementation: 
  scaling_factor = min(1, scaling_factor)

  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next

  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  solver_state.cumulative_kkt_passes += 3

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)
end

"""
Takes a step with constant step size using steering vectors.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.

Different version of DWIFOB:
  - This one where we combine the alt_A and alt_B versions. 
"""
function take_dwifob_step_alt_C(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
  debugging=false
)
  # Initializing the hat variables of the algorithm:
  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)

  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)

  # Update the solver states: 
  solver_state.current_primal_solution = x_next
  solver_state.current_dual_solution = y_next

  # Preparing the input for the Regularized Andersson Acceleration:
  push!(dwifob_solver_state.primal_iterates, p_x_k)
  push!(dwifob_solver_state.dual_iterates, p_y_k)
  if (m_k < dwifob_solver_state.current_iteration) 
    popfirst!(dwifob_solver_state.primal_iterates)
    popfirst!(dwifob_solver_state.dual_iterates)
    popfirst!(dwifob_solver_state.x_hat_iterates)
    popfirst!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k (linear combination of inertial terms) for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  u_hat_x_deviation_sum = zeros(size(x_k))
  u_hat_y_deviation_sum = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    u_hat_x_deviation_sum += primal_i * alpha_i
    u_hat_y_deviation_sum += dual_i * alpha_i
  end 
  u_hat_x_next = -p_x_k + u_hat_x_deviation_sum
  u_hat_y_next = -p_y_k + u_hat_y_deviation_sum

  # Calculating the l^2_k factor: 
  multiplicative_factor = lambda_k * (4 - 2 * lambda_k) * (4 - 2 * lambda_next) / (4 * lambda_next)
  norm_argument = [p_x_k; p_y_k] - [x_k; y_k] + (2 * lambda_k - 2) / (4 - 2 * lambda_k) * [u_x_k; u_y_k]
  l_squared_k = multiplicative_factor * squared_norm_M(norm_argument, problem, solver_state)

  # Calculating the deviations for the next iteration:
  u_next_hat = [u_hat_x_next; u_hat_y_next]
  scaling_factor = dwifob_solver_state.zeta_k * sqrt(l_squared_k) 
  scaling_factor = scaling_factor / (dwifob_solver_state.epsilon + sqrt(squared_norm_M(u_next_hat, problem, solver_state)))
  scaling_factor = min(1, scaling_factor)

  u_x_next = scaling_factor * u_hat_x_next
  u_y_next = scaling_factor * u_hat_y_next

  # Calculating the hat iterates:
  x_hat_next = x_next + u_x_next
  y_hat_next = y_next + u_y_next

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  solver_state.cumulative_kkt_passes += 3

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  dwifob_solver_state.current_primal_deviation = u_x_next
  dwifob_solver_state.current_dual_deviation = u_y_next
  push!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  push!(dwifob_solver_state.y_hat_iterates, y_hat_next)  
end

"""
  An inertial variant of PDHG, using the DWIFOB struct out of convenience for now. 
"""
function take_inertial_pdhg_step(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
)
  
  # How far back do we remember when calculating the inertial term?
  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  
  # We store the previous "steps" i.e. differences between iterates,
  # in the x_hat_iterates list for convenience for now.
  
  # First we take a regular PDHG step: 
  


  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    push!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    push!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_hat_k = last(dwifob_solver_state.x_hat_iterates)
  y_hat_k = last(dwifob_solver_state.y_hat_iterates)

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  u_x_k = dwifob_solver_state.current_primal_deviation
  u_y_k = dwifob_solver_state.current_dual_deviation

  lambda_k = dwifob_solver_state.lambda_k
  lambda_next = dwifob_solver_state.lambda_next

  if isnan(x_hat_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k)
  K_trans_y_next = problem.constraint_matrix' * y_next

  # Update the solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next,
  )

end