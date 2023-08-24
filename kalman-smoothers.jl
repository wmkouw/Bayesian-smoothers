"""
Implementations of standard kalman smoothers
"""

using Distributions
using Random


function fixedinterval_rts_smoother(observations,
                                    transition_matrix,
                                    emission_matrix,
                                    process_noise,
                                    measurement_noise,
                                    state0)
    """
    Fixed-interval Rauch-Tung-Striebel smoother 
    Bayesian Filtering & Smoothing (Särkka, 2014), Th. 8.2

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
    mk = zeros(Dx, time_horizon)
    Pk = zeros(Dx, Dx, time_horizon)

    # Initial state prior
    m_0, P_0 = state0

    # Start previous state variable
    m_kmin = m_0
    P_kmin = P_0

    "Forward pass"
    for k = 1:time_horizon

        # Forward prediction step
        m_k_pred = transition_matrix*m_kmin
        P_k_pred = transition_matrix*P_kmin*transition_matrix' .+ process_noise

        # Forward update step
        v_k = observations[:,k] .- emission_matrix*m_k_pred
        S_k = emission_matrix*P_k_pred*emission_matrix' .+ measurement_noise
        K_k = P_k_pred*emission_matrix'*inv(S_k)
        m_k = m_k_pred .+ K_k*v_k
        P_k = P_k_pred .- K_k*S_k*K_k'
        
        # Store estimates
        mk[:,k] = m_k
        Pk[:,:,k] = P_k

        # Update previous state variable
        m_kmin = m_k
        P_kmin = P_k

    end

    # Initialize smoothing estimate arrays
    msk = zeros(Dx, time_horizon)
    Psk = zeros(Dx, Dx, time_horizon)

    # Smoothed estimates at time horizon
    msk[:,time_horizon] = mk[:,time_horizon]
    Psk[:,:,time_horizon] = Pk[:,:,time_horizon]

    "Backward pass"
    for k = time_horizon-1:-1:1
        
        # Backward prediction
        m_kplus = transition_matrix * mk[:,k]
        P_kplus = transition_matrix * Pk[:,:,k] * transition_matrix' .+ process_noise

        # Backward update step
        G_k = Pk[:,:,k]*transition_matrix' * inv(P_kplus)
        msk[:,k] = mk[:,k] + G_k*(msk[:,k+1] - m_kplus)
        Psk[:,:,k] = Pk[:,:,k] + G_k*(Psk[:,:,k+1] - P_kplus)*G_k'

    end

    return msk, Psk
end
