"""
Implementation of a Rauch-Tung-Striebel smoother based on Ch.8 of Bayesian Filtering & Smoothing (Särkka, 2014)

Wouter Kouw
03-07-2021
"""

using Distributions
using Random

include("util.jl")

function rts_smoother(observations,
                       transition_matrix,
                       emission_matrix,
                       process_noise,
                       measurement_noise,
                       state0)
    """
    Rauch-Tung-Striebel smoother (Th. ?.?)

    This filter is built for a linear Gaussian dynamical system with known
    transition coefficients, process and measurement noise.
    """

    # Dimensionality
    Dx = size(process_noise,1)
    Dy = size(measurement_noise,1)

    # Recast process noise to matrix
    if Dx == 1
        if typeof(process_noise) != Array{Float64,2}
            process_noise = reshape([process_noise], 1, 1)
        end
        if typeof(measurement_noise) != Array{Float64,2}
            measurement_noise = reshape([measurement_noise], 1, 1)
        end
    end

    # Time horizon
    time_horizon = length(observations)

    # Initialize estimate arrays
    mx = zeros(Dx, time_horizon)
    Px = zeros(Dx, Dx, time_horizon)

    # Initial state prior
    m_0, P_0 = state0

    # Start previous state variable
    m_tmin = m_0
    P_tmin = P_0

    for t = 1:time_horizon

        # Prediction step
        m_t_pred = transition_matrix*m_tmin
        P_t_pred = transition_matrix*P_tmin*transition_matrix' .+ process_noise

        # Update step
        v_t = observations[:,t] .- emission_matrix*m_t_pred
        S_t = emission_matrix*P_t_pred*emission_matrix' .+ measurement_noise
        K_t = P_t_pred*emission_matrix'*inv(S_t)
        m_t = m_t_pred .+ K_t*v_t
        P_t = P_t_pred .- K_t*S_t*K_t'

        # Store estimates
        mx[:,t] = m_t
        Px[:,:,t] = P_t

        # Update previous state variable
        m_tmin = m_t
        P_tmin = P_t
    end
    return mx, Px
end
