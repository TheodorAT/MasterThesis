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

mutable struct SteeringVectorSolverState
  """Hyperparameters to the steering vector method""" 
  lambda_0::Float64
  lambda_k::Float64
  kappa::Float64   
  beta_bar::Float64

  """Current iteration of the steering vectors"""
  current_iteration::Int64

  """The primal and dual hat iteratess"""
  prev_hat_primal::Vector{Float64}
  prev_hat_dual::Vector{Float64}

  """Gradient iterates"""
  prev_primal_gradient::Vector{Float64}
  prev_dual_gradient::Vector{Float64}

  """Deviation/Steering vector iterates"""
  current_primal_deviation::Vector{Float64}
  current_dual_deviation::Vector{Float64}

  """Movement iterates"""
  prev_movement::Vector{Float64}

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

function compute_next_dual_solution_no_dual_product(
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
  return next_dual
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
    steering_vector_solver_state = initialize_steering_vector_state(primal_size, dual_size)
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
          steering_vector_solver_state = initialize_steering_vector_state(primal_size, dual_size)
        end
      end
      if current_iteration_stats.restart_used == RESTART_CHOICE_RESTART_TO_AVERAGE
        solver_state.current_dual_product =
        problem.constraint_matrix' * solver_state.current_dual_solution
        # println("Restarted to average.")
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
        steering_vector_solver_state = initialize_steering_vector_state(primal_size, dual_size)
      end
    end

    #### The restart section for constant restarts of dwifob
    if (iteration > 10 && mod(iteration - 1, params.dwifob_restart_frequency) == 0 )
      if (params.dwifob_restart == "constant")
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
        steering_vector_solver_state = initialize_steering_vector_state(primal_size, dual_size)
      elseif (params.dwifob_restart == "NOFOB")
        new_x = dwifob_solver_state.current_u_hat_x_deviation_sum
        new_y = dwifob_solver_state.current_u_hat_y_deviation_sum
        dwifob_solver_state, dwifob_matrix_cache = initialize_dwifob_state(dwifob_params, primal_size, dual_size, dwifob_solver_state.maximum_singular_value)
        steering_vector_solver_state = initialize_steering_vector_state(primal_size, dual_size)
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
      elseif params.dwifob_option == "inertial_PDHG"
        take_inertial_pdhg_step(params.step_size_policy_params, problem, solver_state, dwifob_solver_state)
      elseif params.dwifob_option == "AA_PDHG"
        take_anderson_pdhg_step2(params.step_size_policy_params, problem, solver_state, dwifob_solver_state)
      elseif params.dwifob_option == "momentum_steering"
        take_momentum_steering_step(params.step_size_policy_params, problem, solver_state, steering_vector_solver_state)
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


function try_feasibility_polishing(

)
  return
  # Construct the averages of the primal and dual solutions since last restart.

  # Check if the objective gap is sufficiently small to do feasibility polishing
  
  # If it is, then we can do feasibility polishing.
  ### Primal feasibility polishing 

  # If we terminated the primal feasibility polishing without finding a optimal
  # solution, we also return, 
  # since we don't want to continue the feasibility polishing, 
  # intead we return to regular iterations.

  ### Dual feasibility polishing

  # If we successfully terminated the dual feasibility polishing, 
  # we return to regular iterations. And we continue the algorithm.
end

function primal_polishing(
  primal_solution::Vector{Float64},
)
  # Make copy of of the solver state, but with objective function as 0.
  return
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

"""Initializes the state for the structs required in dwifob."""
function initialize_steering_vector_state(
  primal_size::Int64,
  dual_size::Int64,
)
  lambda_0 = 1
  kappa = 0.9
  beta_bar = 0
  steering_vector_solver_state = SteeringVectorSolverState(
    lambda_0,                 # lambda_0 
    lambda_0,                 # lambda_k
    kappa,                    # kappa
    beta_bar,                 # beta_bar
    0,                        # current iteration since last restart
    zeros(primal_size),       # primal hat iterate
    zeros(dual_size),         # dual hat iterate
    zeros(primal_size),       # primal gradient
    zeros(dual_size),         # dual gradient
    zeros(primal_size),       # primal steering vector
    zeros(dual_size),         # dual steering vector
    zeros(primal_size + dual_size) # The past movement:
  )
  return steering_vector_solver_state
end

### Different implemented steps: ####
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


# TODO: Implement this method with the step sizes as picked in PDLP. 
"""
Takes a step with constant step size using steering vectors, selected in an inertial direction.
Modifies the third and fourth arguments: solver_state and dwifob_solver_state.
"""
function take_momentum_steering_step(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  steering_vector_solver_state::SteeringVectorSolverState,
  debugging=false
)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  
  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution
  tau = solver_state.step_size / solver_state.primal_weight
  sigma = solver_state.step_size * solver_state.primal_weight

  # Initializing the state of the algorithm:
  if (steering_vector_solver_state.current_iteration == 0)
    steering_vector_solver_state.prev_hat_primal = x_k
    steering_vector_solver_state.prev_hat_dual = y_k
    steering_vector_solver_state.prev_primal_gradient = x_k
    steering_vector_solver_state.prev_dual_gradient = y_k
  end
  
  # Previous iterates
  x_hat_prev = steering_vector_solver_state.prev_hat_primal
  y_hat_prev = steering_vector_solver_state.prev_hat_dual
  p_x_prev = steering_vector_solver_state.prev_primal_gradient
  p_y_prev = steering_vector_solver_state.prev_dual_gradient

  # The deviation vectors:
  u_x_k = steering_vector_solver_state.current_primal_deviation
  u_y_k = steering_vector_solver_state.current_dual_deviation

  # Deviation vector hyperparameters:
  lambda_0 = steering_vector_solver_state.lambda_0
  lambda_k = steering_vector_solver_state.lambda_k
  kappa = steering_vector_solver_state.kappa        # This value can change the results a lot it seems like. 
  beta_bar = steering_vector_solver_state.beta_bar  # Maybe change this to something really small instead.

  if isnan(x_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the hat iterates:
  x_hat_k = x_k + ((lambda_k - lambda_0)/lambda_k) * (x_hat_prev - x_k) + u_x_k  
  y_hat_k = y_k + ((lambda_k - lambda_0)/lambda_k) * (y_hat_prev - y_k) + u_y_k

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
  p_x_k = x_hat_k - tau * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
  p_y_k = y_hat_k + sigma * dual_gradient
  project_dual!(p_y_k, problem)

  # Calculating the next iterates:
  x_next = x_k + lambda_k * (p_x_k - x_hat_k) + (lambda_k  - lambda_0) * (x_hat_prev - p_x_prev)
  y_next = y_k + lambda_k * (p_y_k - y_hat_k) + (lambda_k  - lambda_0) * (y_hat_prev - p_y_prev)
  
  # Calculating the deviation for the next iterates: 
  cur_movement = [(x_next - x_k); (y_next - y_k)]
  prev_movement = steering_vector_solver_state.prev_movement
  
  step_similarity = calculate_step_similarity(cur_movement, prev_movement)
  similarity_threshold = 0.9
  if (steering_vector_solver_state.current_iteration > 0 && step_similarity >= similarity_threshold)
    # TODO: Ask if the primal step size really should be used like this here? 
    mult_factor_all = kappa * (4 - tau * beta_bar - 2 * lambda_0)/2
    mult_factor_u_cur = (2 - tau * beta_bar  - 2 * lambda_0)/(4 - tau * beta_bar - 2 * lambda_0) 
    u_x_next = mult_factor_all * (p_x_k - x_k + ((lambda_k - lambda_0)/lambda_k) * (x_k - p_x_prev) -  mult_factor_u_cur * u_x_k)
    u_y_next = mult_factor_all * (p_y_k - y_k + ((lambda_k - lambda_0)/lambda_k) * (y_k - p_y_prev) -  mult_factor_u_cur * u_y_k)
  else
    u_x_next = 0 .* u_x_k
    u_y_next = 0 .* u_y_k
  end

  # Storing the variables for the next iteration:
  K_trans_y_next = problem.constraint_matrix' * y_next
  # The regular solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next,
  )
  # The steering vector solver state:
  steering_vector_solver_state.current_iteration += 1 
  # Steering vectors:
  steering_vector_solver_state.current_primal_deviation = u_x_next
  steering_vector_solver_state.current_dual_deviation = u_y_next
  # Hat iterates:  
  steering_vector_solver_state.prev_hat_primal = x_hat_k
  steering_vector_solver_state.prev_hat_dual = y_hat_k
  # Gradient iterates:
  steering_vector_solver_state.prev_primal_gradient = p_x_k
  steering_vector_solver_state.prev_dual_gradient = p_y_k
  # Movement:
  steering_vector_solver_state.prev_movement = cur_movement  
end

"""
Takes a step with adaptive step size using steering vectors, selected in an inertial direction.
Modifies the third and fourth arguments: solver_state and steering_vector_solver_state.
"""
function take_momentum_steering_step(
  step_params::AdaptiveStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  steering_vector_solver_state::SteeringVectorSolverState,
  debugging=false
)
  step_size = solver_state.step_size
  done = false
  iter = 0

  while !done
    iter += 1
    solver_state.total_number_iterations += 1

    # Extracting some variables from the solver state struct
    # for clearer and more concise code:  
    x_k = solver_state.current_primal_solution
    y_k = solver_state.current_dual_solution
    tau = step_size / solver_state.primal_weight
    sigma = step_size * solver_state.primal_weight

    # Initializing the state of the algorithm:
    if (steering_vector_solver_state.current_iteration == 0 && iter == 1)
      steering_vector_solver_state.prev_hat_primal = x_k
      steering_vector_solver_state.prev_hat_dual = y_k
      steering_vector_solver_state.prev_primal_gradient = x_k
      steering_vector_solver_state.prev_dual_gradient = y_k
    end
    
    # Previous iterates
    x_hat_prev = steering_vector_solver_state.prev_hat_primal
    y_hat_prev = steering_vector_solver_state.prev_hat_dual
    p_x_prev = steering_vector_solver_state.prev_primal_gradient
    p_y_prev = steering_vector_solver_state.prev_dual_gradient

    # The deviation vectors:
    u_x_k = steering_vector_solver_state.current_primal_deviation
    u_y_k = steering_vector_solver_state.current_dual_deviation

    # Deviation vector hyperparameters:
    lambda_0 = steering_vector_solver_state.lambda_0
    lambda_k = steering_vector_solver_state.lambda_k
    kappa = steering_vector_solver_state.kappa        # This value can change the results a lot it seems like. 
    beta_bar = steering_vector_solver_state.beta_bar  # Maybe change this to something really small instead.

    if isnan(x_k[1]) 
      println("Got NaN in iterates, aborting...")
      exit(1)
    end

    # Calculating the hat iterates:
    x_hat_k = x_k + ((lambda_k - lambda_0)/lambda_k) * (x_hat_prev - x_k) + u_x_k  
    y_hat_k = y_k + ((lambda_k - lambda_0)/lambda_k) * (y_hat_prev - y_k) + u_y_k

    # Calculating the primal "pseudogradient" (p_x_k) value:
    primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_hat_k
    p_x_k = x_hat_k - tau * primal_gradient
    project_primal!(p_x_k, problem)

    # Calculating the dual "pseudogradient" (p_y_k) value: 
    dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_hat_k)
    p_y_k = y_hat_k + sigma * dual_gradient
    project_dual!(p_y_k, problem)

    # Calculating the next iterates:
    x_next = x_k + lambda_k * (p_x_k - x_hat_k) + (lambda_k  - lambda_0) * (x_hat_prev - p_x_prev)
    y_next = y_k + lambda_k * (p_y_k - y_hat_k) + (lambda_k  - lambda_0) * (y_hat_prev - p_y_prev)

    if isnan(y_next[1]) 
      println("Got NaN in iterates, aborting...")
      exit(1)
    end

    next_dual_product = problem.constraint_matrix' * y_next
    interaction, movement = compute_interaction_and_movement(
      solver_state,
      problem,
      x_next,
      y_next,
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
      done = true
      update_solution_in_solver_state(
        solver_state,
        x_next,
        y_next,
        next_dual_product,
      )
        # Calculating the deviation for the next iterates: 
      cur_movement = [(x_next - x_k); (y_next - y_k)]
      prev_movement = steering_vector_solver_state.prev_movement
      
      step_similarity = calculate_step_similarity(cur_movement, prev_movement)
      similarity_threshold = -2
      if (steering_vector_solver_state.current_iteration > 0 && step_similarity >= similarity_threshold)
        # TODO: Ask if the primal step size really should be used like this here? 
        mult_factor_all = kappa * (4 - tau * beta_bar - 2 * lambda_0)/2
        mult_factor_u_cur = (2 - tau * beta_bar  - 2 * lambda_0)/(4 - tau * beta_bar - 2 * lambda_0) 
        u_x_next = mult_factor_all * (p_x_k - x_k + ((lambda_k - lambda_0)/lambda_k) * (x_k - p_x_prev) -  mult_factor_u_cur * u_x_k)
        u_y_next = mult_factor_all * (p_y_k - y_k + ((lambda_k - lambda_0)/lambda_k) * (y_k - p_y_prev) -  mult_factor_u_cur * u_y_k)
      else
        u_x_next = 0 .* u_x_k
        u_y_next = 0 .* u_y_k
      end

      # Storing the variables for the next iteration:
      K_trans_y_next = problem.constraint_matrix' * y_next
      # The regular solver state: 
      update_solution_in_solver_state(
        solver_state,
        x_next,
        y_next,
        K_trans_y_next,
      )
      # The steering vector solver state:
      steering_vector_solver_state.current_iteration += 1 
      # Steering vectors:
      steering_vector_solver_state.current_primal_deviation = u_x_next
      steering_vector_solver_state.current_dual_deviation = u_y_next
      # Hat iterates:  
      steering_vector_solver_state.prev_hat_primal = x_hat_k
      steering_vector_solver_state.prev_hat_dual = y_hat_k
      # Gradient iterates:
      steering_vector_solver_state.prev_primal_gradient = p_x_k
      steering_vector_solver_state.prev_dual_gradient = p_y_k
      # Movement:
      steering_vector_solver_state.prev_movement = cur_movement
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

function should_use_inertia(
  x_cur::Vector{Float64}, 
  y_cur::Vector{Float64},
  x_next::Vector{Float64}, 
  y_next::Vector{Float64},
  dwifob_solver_state::DwifobSolverState,
) 
  cur_movement = [(x_next - x_cur); (y_next - y_cur)]
  prev_movement = [first(dwifob_solver_state.x_hat_iterates); first(dwifob_solver_state.y_hat_iterates)]
  
  # If the past two steps are of sufficiently similar direction, we want to add inertia, otherwise not.  
  cur_movement_unit = cur_movement / norm(cur_movement, 2)
  prev_movement_unit = prev_movement / norm(prev_movement, 2)

  similarity = cur_movement_unit' * prev_movement_unit  
  # println("Similarity: ", similarity)
  # TODO: Perhaps we should use the (weighted) average similarity over multiple iterations?  
  # TODO: Maybe return the similarity, and use that as a weight for the inertial terms as well.
  # This threshold decides when we should use momentum.
  similarity_threshold = 0.90
  return similarity >= similarity_threshold
end


function calculate_step_similarity(
  cur_movement,
  prev_movement,
) 
  # If the past two steps are of sufficiently similar direction, we want to add inertia, otherwise not.  
  cur_movement_unit = cur_movement / norm(cur_movement, 2)
  prev_movement_unit = prev_movement / norm(prev_movement, 2)
  similarity = cur_movement_unit' * prev_movement_unit  
  return similarity
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
  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  if isnan(x_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  # Calculating the primal "pseudogradient" (p_x_k) value:
  primal_gradient = problem.objective_vector - problem.constraint_matrix' * y_k
  p_x_k = x_k - (solver_state.step_size / solver_state.primal_weight) * primal_gradient
  project_primal!(p_x_k, problem)

  # Calculating the dual "pseudogradient" (p_y_k) value: 
  dual_gradient = problem.right_hand_side - problem.constraint_matrix * (2 * p_x_k - x_k)
  p_y_k = y_k + (solver_state.step_size * solver_state.primal_weight) * dual_gradient
  project_dual!(p_y_k, problem)
  
  # Calculating the next iterates:
  if (m_k != 0 && should_use_inertia(x_k, y_k, p_x_k, p_y_k, dwifob_solver_state))  
    # Calculating the inertial terms: 
    inertial_term_x, inertial_term_y = calculate_inertia(dwifob_solver_state)    
    x_next = p_x_k + inertial_term_x
    y_next = p_y_k + inertial_term_y
  else 
    x_next = p_x_k
    y_next = p_y_k    
  end 
  
  K_trans_y_next = problem.constraint_matrix' * y_next

  # Update the solver state: 
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    K_trans_y_next,
  )

  # Saving the "movement" from this step in the dwifob solver struct.
  movement_x = x_next - x_k
  movement_y = y_next - y_k  
  pushfirst!(dwifob_solver_state.x_hat_iterates, movement_x)
  pushfirst!(dwifob_solver_state.y_hat_iterates, movement_y)

  # If "memory is full", we forget the movement from the least recent iteration: 
  if (dwifob_solver_state.current_iteration > m_k) 
    pop!(dwifob_solver_state.x_hat_iterates)
    pop!(dwifob_solver_state.y_hat_iterates)
  end

  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 1 times, and K^T 1 times. (1 each for steps) 
  solver_state.cumulative_kkt_passes += 1

  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
end

"""
  An inertial variant of PDHG with adaptive step size, using the DWIFOB struct out of convenience for now. 
"""
# TODO: Should we do proj after inertia, probably. 
# Especially if the largest inertia losers have a lot of single contraints, that are affected by proj.  
function take_inertial_pdhg_step(
  step_params::AdaptiveStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
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

    next_dual = compute_next_dual_solution_no_dual_product(
      problem,
      solver_state.current_primal_solution,
      next_primal,
      solver_state.current_dual_solution,
      step_size,
      solver_state.primal_weight,
    )

    ##### Here we should add inertial terms: ##### 
    # How far back do we remember when calculating the inertial term?
    m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
    
    # We store the previous "steps" i.e. differences between iterates,
    # in the x_hat_iterates list for convenience for now.
    x_k = solver_state.current_primal_solution
    y_k = solver_state.current_dual_solution

    if isnan(x_k[1]) 
      println("Got NaN in iterates, aborting...")
      exit(1)
    end

      # Calculating the next iterates:
    if (m_k > 0 && should_use_inertia(x_k, y_k, next_primal, next_dual, dwifob_solver_state))  
      # Calculating the inertial terms: 
      inertial_term_x, inertial_term_y = calculate_inertia(dwifob_solver_state)    
      next_primal = next_primal + inertial_term_x
      next_dual = next_dual + inertial_term_y
    end
    next_dual_product = problem.constraint_matrix' * next_dual

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
      done = true
      update_solution_in_solver_state(
        solver_state,
        next_primal,
        next_dual,
        next_dual_product,
      )

      # Saving the "movement" from this step in the dwifob solver struct.
      movement_x = next_primal - x_k
      movement_y = next_dual - y_k  
      pushfirst!(dwifob_solver_state.x_hat_iterates, movement_x)
      pushfirst!(dwifob_solver_state.y_hat_iterates, movement_y)

      # If "memory is full", we forget the movement from the least recent iteration: 
      if (dwifob_solver_state.current_iteration > m_k) 
        pop!(dwifob_solver_state.x_hat_iterates)
        pop!(dwifob_solver_state.y_hat_iterates)
      end

      # Updating the changes in the mutable dwifob struct before the next iteration:
      dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
    end

    first_term = (step_size_limit * 
      (1 - (solver_state.total_number_iterations + 1)^(-step_params.reduction_exponent)))
    second_term = (step_size * 
      (1 + (solver_state.total_number_iterations + 1)^(-step_params.growth_exponent)))
    step_size = min(first_term, second_term)
  end
  solver_state.step_size = step_size
end

function calculate_inertia(
  dwifob_solver_state::DwifobSolverState,
)
  # Hyperparameters: (this will grow haha)
  beta = 0.5        # Controls the weight of past steps, more frequent have more impact.
  dampening = 0.4   # Controls the size of the momentum term in relation to the steps. 
                    # (Very reasonable to keep this below 0.5)

  # Calculate the average movement from the last iterates (stored in hat_iterates for this purpose):
  vector_x = zeros(size(last(dwifob_solver_state.x_hat_iterates)))
  vector_y = zeros(size(last(dwifob_solver_state.y_hat_iterates)))
  count = 0
  for (step_x, step_y) in zip(dwifob_solver_state.x_hat_iterates, dwifob_solver_state.y_hat_iterates)  
    vector_x += step_x * (beta^count)
    vector_y += step_y * (beta^count)
    count += 1
  end  
  vector_x = vector_x * dampening / count
  vector_y = vector_y * dampening / count
  
  return vector_x, vector_y
end

"""
  An Anderson Accelerated variant of PDHG, using the DWIFOB struct out of convenience for now. 
"""
function take_anderson_pdhg_step(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
)
  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    pushfirst!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    pushfirst!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  step_size = solver_state.step_size
  primal_weight = solver_state.primal_weight

  if isnan(x_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  g_x_k = compute_next_primal_solution(
    problem,
    solver_state.current_primal_solution,
    solver_state.current_dual_product,
    step_size,
    primal_weight,
  )

  g_y_k = compute_next_dual_solution_no_dual_product(
    problem,
    solver_state.current_primal_solution,
    g_x_k,
    solver_state.current_dual_solution,
    step_size,
    primal_weight,
  )

  # The AA part:
  pushfirst!(dwifob_solver_state.primal_iterates, g_x_k)
  pushfirst!(dwifob_solver_state.dual_iterates, g_y_k)
  if (m_k < dwifob_solver_state.current_iteration) 
    pop!(dwifob_solver_state.primal_iterates)
    pop!(dwifob_solver_state.x_hat_iterates)
    pop!(dwifob_solver_state.dual_iterates)
    pop!(dwifob_solver_state.y_hat_iterates)
  end

  # Calculating R_k for the RAA:   
  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  # Converting to matrixes for easier calculations: 
  R_k_primal = reduce(hcat, R_k_primal)
  R_k_dual = reduce(hcat, R_k_dual)
  # Finally, forming R_k and calculating the RAA:
  R_k = vcat(R_k_primal, R_k_dual)
  alpha_k = calculate_anderson_acceleration(solver_state, R_k, m_k)

  # Calculating the hat_deviations: 
  x_next = zeros(size(x_k))
  y_next = zeros(size(y_k))
  for (primal_i, dual_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, dwifob_solver_state.dual_iterates, alpha_k)
    x_next += primal_i * alpha_i
    y_next += dual_i * alpha_i
  end

  next_dual_product = problem.constraint_matrix' * y_next
  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    next_dual_product,
  )
  # Each iteration of this dwifob implementation calculates a matrix calculation 
  # using the constraint matrix: K: 3 times, and K^T 3 times. (1 each for steps, 1 each for the 2 M-norms) 
  solver_state.cumulative_kkt_passes += 1
  
  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  pushfirst!(dwifob_solver_state.x_hat_iterates, x_next)
  pushfirst!(dwifob_solver_state.y_hat_iterates, y_next)  
end

"""
  An Anderson Accelerated variant of PDHG, using the DWIFOB struct out of convenience for now. 
"""
function take_anderson_pdhg_step2(
  step_params::ConstantStepsizeParams,
  problem::QuadraticProgrammingProblem,
  solver_state::PdhgSolverState,
  dwifob_solver_state::DwifobSolverState,
)
  # Initializing the hat variables of the algorithm:
  if (dwifob_solver_state.current_iteration == 0)
    pushfirst!(dwifob_solver_state.x_hat_iterates, solver_state.current_primal_solution)
    pushfirst!(dwifob_solver_state.y_hat_iterates, solver_state.current_dual_solution)
  end

  m_k = min(dwifob_solver_state.max_memory, dwifob_solver_state.current_iteration)
  # Extracting some variables from the solver state struct
  # for clearer and more concise code:  

  x_k = solver_state.current_primal_solution
  y_k = solver_state.current_dual_solution

  step_size = solver_state.step_size
  primal_weight = solver_state.primal_weight

  if isnan(x_k[1]) 
    println("Got NaN in iterates, aborting...")
    exit(1)
  end

  primal_gradient = problem.objective_vector .- solver_state.current_dual_product
  g_x_k = x_k .- (step_size / primal_weight) .* primal_gradient
  
  # The AA for the primal iterates:
  pushfirst!(dwifob_solver_state.primal_iterates, g_x_k)
  if (m_k < dwifob_solver_state.current_iteration) 
    pop!(dwifob_solver_state.primal_iterates)
    pop!(dwifob_solver_state.x_hat_iterates)
  end

  R_k_primal = dwifob_solver_state.primal_iterates - dwifob_solver_state.x_hat_iterates
  R_k_primal = reduce(hcat, R_k_primal)
  alpha_k_primal = calculate_anderson_acceleration(solver_state, R_k_primal, m_k)

  # Calculating the next primal iterates from the AA: 
  x_hat_next = zeros(size(x_k))
  for (primal_i, alpha_i) in zip(dwifob_solver_state.primal_iterates, alpha_k_primal)
    x_hat_next = x_hat_next + (primal_i * alpha_i)
  end 
  x_next = copy(x_hat_next)
  project_primal!(x_next, problem)

  dual_gradient = problem.right_hand_side .- problem.constraint_matrix * (2 .* x_next - x_k)
  g_y_k = y_k .+ (primal_weight * step_size) .* dual_gradient
  
  # The AA for the dual iterates: 
  pushfirst!(dwifob_solver_state.dual_iterates, g_y_k)
  if (m_k < dwifob_solver_state.current_iteration) 
    pop!(dwifob_solver_state.dual_iterates)
    pop!(dwifob_solver_state.y_hat_iterates)
  end
  R_k_dual = dwifob_solver_state.dual_iterates - dwifob_solver_state.y_hat_iterates
  R_k_dual = reduce(hcat, R_k_dual)
  alpha_k_dual = calculate_anderson_acceleration(solver_state, R_k_dual, m_k)

  # Calculating the next primal iterates from the AA: 
  y_hat_next = zeros(size(y_k))
  for (dual_i, alpha_i) in zip(dwifob_solver_state.dual_iterates, alpha_k_dual)
    y_hat_next = y_hat_next + (dual_i * alpha_i)
  end 
  y_next = copy(y_hat_next)
  project_dual!(y_next, problem)
  next_dual_product = problem.constraint_matrix' * y_next

  update_solution_in_solver_state(
    solver_state,
    x_next,
    y_next,
    next_dual_product,
  )
  # Each iteration of this dwifob implementation calculates a matrix calculation 
  solver_state.cumulative_kkt_passes += 1
  
  # Updating the changes in the mutable dwifob struct before the next iteration:
  dwifob_solver_state.current_iteration = dwifob_solver_state.current_iteration + 1
  pushfirst!(dwifob_solver_state.x_hat_iterates, x_hat_next)
  pushfirst!(dwifob_solver_state.y_hat_iterates, y_hat_next)  
end

