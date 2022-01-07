"""
Implementation of a fixed-lag smoother based on Wikipedia 
(https://en.wikipedia.org/wiki/Kalman_filter#Fixed-lag_smoother) and
Roger Labbe Jr.'s filterpy package (https://filterpy.readthedocs.io/en/latest/kalman/FixedLagSmoother.html)

Author: Wouter Kouw
Last update: 06-01-2022
"""

using Distributions
using LinearAlgebra
using Random

function fixedlag_rts_smoother(y, A, C, Q, R, x0; N=1)
    """
    Fixed-lag Rauch-Tung-Striebel smoother (Th. 8.2)

    This filter is built for a linear Gaussian dynamical system with known
    transition coefficients, process and measurement noise.
    """

    # Dimensionality
    Dx = size(Q,1)
    Dy = size(R,1)

    # Define identity matrix
    Ix = Matrix{Float64}(I,Dx,Dx)

    # Recast process noise to matrix
    if Dx == 1
        if typeof(Q) != Array{Float64,2}
            Q = reshape([Q], 1, 1)
        end
    end
    if Dy == 1
        if typeof(R) != Array{Float64,2}
            R = reshape([R], 1, 1)
        end
    end

    # Time horizon
    T = length(y)

    # Initialize estimate arrays
    mk = zeros(Dx, T)
    Pk = zeros(Dx, Dx, T)

    # Start previous state variable
    m_kmin1 = x0[1]
    P_kmin1 = x0[2]

    for k in 1:T

        # Predict step
        m_pred = A*m_kmin1
        P_pred = A*P_kmin1*A' + Q

        # Update step
        v = y[k] .- C*m_pred
        S = C*P_pred*C' + R
        SI = inv(S)
        K = P_pred*C'*SI

        # Populate arrays
        m_updt = m_pred + K*v
        P_updt = (Ix - K*C)*P_pred
        mk[:,k] = m_pred
        Pk[:,:,k] = P_updt

        # Compute invariants
        CTSI = C'*SI
        A_KC = (A - K*C)'

        # Start hindsight loop
        if k >= N

            # Initialize smoothed covariance
            PS = P_updt
            for i =1:N

                # Smoothed gain
                K = PS*CTSI
                
                # Smoothed covariance
                PS = PS*A_KC 

                # Hindsight index
                ix = k-i+1

                # Populate smoothed arrays
                mk[:,ix] = mk[:,ix] + K*v
                Pk[:,:,ix] = PS
            end
        else
            mk[:,k] = m_updt
        end

        # Update previous states
        m_kmin1 = mk[:,k]
        P_kmin1 = Pk[:,:,k]
    end
    return mk, Pk
end

