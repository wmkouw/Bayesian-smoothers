"""
Set of functions for generating signals.
"""

function random_walk(process_noise,
                    measurement_noise,
                    mean_state0,
                    precision_state0;
                    time_horizon=100)
    "Generate data according to a random walk"

    # Dimensionality
    d = size(process_noise, 1)

    # Preallocate
    y = zeros(d, time_horizon)
    x = zeros(d, time_horizon+1)

    # Initialize state
    x[:, 1] = sqrt(inv(precision_state0))*randn(d) + mean_state0

    for t = 1:time_horizon

        # Evolve state
        x[:, t+1] = sqrt(process_noise)*randn(d) + x[:, t]

        # Observe
        y[:, t] = sqrt(measurement_noise)*randn(d) + x[:, t+1]

    end
    return y, x
end

function LGDS(A, C, Q, R, x0; T=100)
    """
    Generate data according to a linear Gaussian dynamical system.
    
    Note: assumes univariate observations
    """

    # Dimensionality
    Dx = size(A,1)
    Dy = size(C,1)

    # Check for Dy = 1
    if Dy != 1; error("Assumes univariate observations"); end

    # Preallocate
    x = zeros(Dx,T)
    y = zeros(T,)

    # Initialize state
    x_kmin1 = x0

    for k = 1:T

        # Evolve state
        x[:,k] = A*x_kmin1 .+ cholesky(Q).U*randn(Dx,)

        # Observe
        y[k,:] = C*x[:,k] .+ sqrt(R)*randn(Dy,)[1]

        # Update previous state
        x_kmin1 = x[:,k]

    end
    return y, x
end

function NLGDS(transition_function,
               emission_function,
               process_noise,
               measurement_noise,
               state0_params;
               time_horizon=100)
    "Generate data according to a nonlinear Gaussian dynamical system"

    # Dimensionality
    d = size(process_noise, 1)

    # Preallocate
    x = zeros(d, time_horizon+1)
    y = zeros(d, time_horizon)

    # Prior parameters
    mean0, var0 = state0_params

    # Initialize state
    x[:, 1] = sqrt(var0)*randn(d) .+ mean0

    for t = 1:time_horizon

        # Evolve state
        x[:, t+1] = sqrt(process_noise)*randn(d) .+ transition_function(x[:,t])

        # Observe
        y[:, t] = sqrt(measurement_noise)*randn(d) .+ emission_function(x[:,t+1])

    end
    return y, x
end
