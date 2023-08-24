"""
Simultaneous estimation of states and other latent variables 
(e.g., parameters, coefficients, noise) through variational Bayesian inference.
"""

using Distributions
using LinearAlgebra
using Random


function AX_smoother(observations::Vector{Vector{Float64}},
    Q::Matrix, C::Matrix, R::Matrix,
    ard_prior::Vector{Gamma{Float64}}, 
    state_prior::FullNormal;
    num_iters::Integer=10)
    """
    Ref: Lutinnen (2013). Fast Variational Bayesian Linear State-Space Model. ECML.

    This smoother infers both states and state transition matrix elements.
    """

    T = length(observations)
    D = size(Q,1)

    m0,S0 = params(state_prior)
    α0    = shape.(ard_prior)
    β0    = rate.( ard_prior)

    # Preallocate
    m = zeros(D,T)
    W = cat([diagm(ones(D)) for k in 1:T]...,dims=3)
    S = cat([diagm(ones(D)) for k in 1:T]...,dims=3)
    μ = zeros(D,D)
    Σ = cat([diagm(ones(D)) for d in 1:D]...,dims=3)
    α = ones(D)
    β = ones(D)

    for n = 1:num_iters

        "State estimation"

        AA = zeros(D,D)
        for i in 1:D
            for j in 1:D
                AA[i,j] = sum([μ[i,d]*μ[j,d] + Σ[i,j,d] for d in 1:D]) 
            end
        end

        S_kmin1 = inv(inv(S0) + AA)
        m_kmin1 = S_kmin1*inv(S0)*m0

        Ψ_diag = diagm(ones(D)) + AA + C'*inv(R)*C
        Ψ_offd = -μ'

        # Forward pass
        for k = 1:T-1
            v_k = C*inv(R)*observations[k]
            W[:,:,k] = S_kmin1*Ψ_offd
            S[:,:,k] = inv(Ψ_diag - W[:,:,k]'*Ψ_offd)
            m[:,k] = S[:,:,k]*(v_k - W[:,:,k]'*m_kmin1)

            S_kmin1 = S[:,:,k]
            m_kmin1 = m[:,k]    
        end

        # Update for final step
        v_k = C*inv(R)*observations[T]
        W[:,:,T] = S_kmin1*Ψ_offd
        S[:,:,T] = inv(diagm(ones(D)) + C'*inv(R)*C - W[:,:,T]'*Ψ_offd)
        m[:,T] = S[:,:,T]*(v_k - W[:,:,T]'*m_kmin1)

        # Backward pass
        U = cat([diagm(ones(D)) for k in 1:T]...,dims=3)
        for k = T-1:-1:1
            U[:,:,k+1] = -W[:,:,k+1]*S[:,:,k+1]
            S[:,:,k] = S[:,:,k] - W[:,:,k+1]*U[:,:,k+1]'
            m[:,k] = m[:,k] - W[:,:,k+1]*m[:,k+1]
        end

        "Parameter estimation"

        # Update relevance variables
        for d in 1:D
            α[d] = α0[d] + D/2
            β[d] = β0[d] + 1/2*sum([μ[j,d]^2 + Σ[j,j,d] for j = 1:D])
        end

        # Update state transition matrix
        for d in 1:D
            Σ[:,:,d] = inv(diagm(α./β) + (m0*m0'+S0) + sum([m[:,k]*m[:,k]' + S[:,:,k] for k in 1:T-1]))
            μ[d,:] = Σ[:,:,d]*sum(cat([[m[d,1]*m0]; [m[d,k]*m[:,k-1] for k in 2:T]]...,dims=2),dims=2)[:,1]
        end
    end
    return m,S,μ,Σ,α,β
end