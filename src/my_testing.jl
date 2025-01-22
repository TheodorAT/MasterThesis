# This is an translation of the mathematical definition of the DWIFOB algorithm into code. 


function dwifob_optimize(

)

# Implementation idea, the things we need for the DWIFOB should be 
# stored in a separate struct.
mutable struct DwifobSolverState
    
    """
        The last m_n iterates of the primal, used in the RAA in DWIFOB.
        Stores x_{n}, x_{n-1}, ... , x_{n-m_n}
    """
    last_primal_iterates::Vector{Float64}

    """
        The last m_n iterates of the primal^{hat}, used in the RAA in DWIFOB.
        Stores x^{hat}_{n}, x^{hat}_{n-1}, ... , x^{hat}_{n-m_n}
    """
    last_primal_hat_iterates::Vector{Float64}
    
    """
        The last m_n iterates of the dual, used in the RAA in DWIFOB.
        Stores my_{n}, my_{n-1}, ... , my_{n-m_n}
    """
    last_dual_iterates::Vector{Float64}

    """
    The last m_n iterates of the dual^{hat}, used in the RAA in DWIFOB.
    Stores my^{hat}_{n}, my^{hat}_{n-1}, ... , my^{hat}_{n-m_n}
    """
    last_dual_hat_iterates::Vector{Float64}
end

function take_dwifob_step(
    step_params::ConstantStepsizeParams,
    problem::QuadraticProgrammingProblem,
    solver_state::PdhgSolverState,
    dwifob_solver_state::DwifobSolverState,
)
" Pseudo-code implementation:
    # How far back do we remember, hyperparameter m.    
    m_n = min(m, n)
    
    # Calculating the pseudo-gradients(?) using the hat primal and dual iterates.  
    p_x_n = J_{tau, A} (x_hat_n - tau * L_star * mu_hat_n - tau * C * x_hat_n)
    p_my_n = J_{sigma, B^{-1}} (my_hat_n + sigma * L * (2 * p_x_n - x_hat_n))

    # Update the next iterates using the pseudo-gradients:
    x_next = x_n + lambda_n * (p_x_n - x_hat_n) 
    my_next = my_n + lambda_n * (p_my_n - my_hat_n)
    
    # The remaining part: Calculate the next iterations of the hat primal and dual.
    # Storing the new values of x and my in the DwifobSolverState

    dwifob_solver_state.last_primal_iterates.append() 
    if (m_n == m) dwifob_solver_state.last_primal_iterates.remove_first() 
    dwifob_solver_state.last_dual_iterates.append() 
    if (m_n == m) dwifob_solver_state.last_dual_iterates.remove_first() 

    # Calculating R_n for the RAA: 
    R_n = zeros(Float64, 2, m_n)
    for (int i = 0; i < m_n; i++) {
        R_n[i][0] = dwifob_solver_state.last_primal_iterates(i) - dwifob_solver_state.last_primal_hat_iterates(i)
        R_n[i][1] = dwifob_solver_state.last_dual_iterates(i) - dwifob_solver_state.last_dual_hat_iterates(i)
    }

    alpha_n = solve_regularized_andersson_accelleration(R_n, epsilon_n, ?F? )

    ... Continue tomorrow.

    "
