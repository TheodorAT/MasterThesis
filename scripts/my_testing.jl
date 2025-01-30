using Printf

# # Implementation idea, the things we need for the DWIFOB should be 
# # stored in a separate struct.
# mutable struct DwifobSolverState
  
#     """
#         The last m_n iterates of the primal, used in the RAA in DWIFOB.
#         Stores x_{n}, x_{n-1}, ... , x_{n-m_n}
#     """
#     primal_iterates::Vector{Float64}
  
#     """
#         The last m_n iterates of the dual, used in the RAA in DWIFOB.
#         Stores my_{n}, my_{n-1}, ... , my_{n-m_n}
#     """
#     dual_iterates::Vector{Float64}

#     """
#         The last m_n iterates of the primal^{hat}, used in the RAA in DWIFOB.
#         Stores x^{hat}_{n}, x^{hat}_{n-1}, ... , x^{hat}_{n-m_n}
#     """
#     primal_hat_iterates::Vector{Float64}

#     """
#     The last m_n iterates of the dual^{hat}, used in the RAA in DWIFOB.
#     Stores my^{hat}_{n}, my^{hat}_{n-1}, ... , my^{hat}_{n-m_n}
#     """
#     dual_hat_iterates::Vector{Float64}

#     current_primal_deviation::Vector{Float64}
#     current_dual_deviation::Vector{Float64}

#     max_memory::int

#     current_iteration::int
  
#     lambda_k::Float64

#     zeta_k::Float64
# end


# function take_dwifob_step(
#     step_params::ConstantStepsizeParams,
#     problem::QuadraticProgrammingProblem,
#     solver_state::PdhgSolverState,
#     dwifob_solver_state::DwifobSolverState,
#     )
#     k = dwifob_solver_state.current_iteration

#     # Initializing the deviation parts of the algorithm:
#     if (k == 0)
#         dwifob_solver_state.primal_hat_iterates.push(solver_state.current_primal_solution)
#         dwifob_solver_state.dual_hat_iterates.push(solver_state.current_dual_solution)

#         dwifob_solver_state.current_primal_deviation = 0;
#         dwifob_solver_state.current_dual_deviation = 0;

#     memory = min(dwifob_solver_state.max_memory, k)

#     x_hat_k = dwifob_solver_state.primal_hat_iterates.last()
#     y_hat_k = dwifob_solver_state.dual_hat_iterates.last()
      
#     tau = solver_state.step_size / solver_state.primal_weight
#     sigma = solver_state.step_size * solver_state.primal_weight
  
#     primal_gradient = problem.objective_vector .- problem.constraint_matrix' * y_hat_k
#     p_x_k = project_primal!(x_hat_k - tau .* primal_gradient)

#     dual_gradient = problem.right_hand_side .- problem.constraint_matrix * (2 .* p_x_k .- x_hat_k)
#     p_y_k = project_dual!(y_hat_k + sigma * dual_gradient)
  
#     x_next = solver_state.current_primal_solution + dwifob_solver_state.lambda_k .* (p_x_k - x_hat_k)
#     y_next = solver_state.current_dual_solution + dwifob_solver_state.lambda_k .* (p_y_k - y_hat_k)

#     # Update the solver states: 
#     solver_state.current_primal_solution = x_next
#     solver_state.current_primal_solution = x_next
#     dwifob_solver_state.current_iteration = k + 1

#     @printf("Iteration %d in dwifob complete \n", k)
# end    

#     # To make sure that things work before continuing, we comment out the rest, 
#     # and try to make the function work as the vanilla Chambolle-Pock

#     # # Storing the new values of x and my in the DwifobSolverState
#     # dwifob_solver_state.last_primal_iterates.append() 
#     # if (m_n == m) dwifob_solver_state.last_primal_iterates.remove_first() 
#     # dwifob_solver_state.last_dual_iterates.append() 
#     # if (m_n == m) dwifob_solver_state.last_dual_iterates.remove_first() 

#     # # Calculating R_n (linear combination of inertial terms) for the RAA: 
#     # R_n = zeros(Float64, 2, m_n)
#     # for (int i = 0; i < m_n; i++) {
#     #     R_n[i][0] = dwifob_solver_state.last_primal_iterates(i) - dwifob_solver_state.last_primal_hat_iterates(i)
#     #     R_n[i][1] = dwifob_solver_state.last_dual_iterates(i) - dwifob_solver_state.last_dual_hat_iterates(i)
#     # }
#     # # Calculate the steering vectors for the next iteration: 
#     # alpha_k = get_RAA_solution()

function main()
  x1 = [1, 1, 1]
  x2 = [2, 2, 2]
  x3 = [3, 3, 3]

  queue = Vector{Vector{Float64}}()

  push!(queue, x1)
  push!(queue, x2)

  # Printing the elements: 
  print("Queue 1: ")
  for element in queue
    print(element, " ")
  end
  print("\n")

  push!(queue, x3)
  popfirst!(queue)

  # Printing the elements: 
  print("Queue 2: ")
  for element in queue
    print(element, " ")
  end
  print("\n")    
  
  println(last(queue))
end

main()