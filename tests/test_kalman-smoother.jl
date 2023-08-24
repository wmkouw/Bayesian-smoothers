using LinearAlgebra
using Distributions
using Plots

include("./gen_data.jl")
include("./kalman_smoothers.jl")

"""Experimental parameters"""

# Time horizon
T = 100

# Experimental parameters
transition_coeffs = 1.0
emission_coeffs = 1.0

# Noises
process_noise = 0.5
measurement_noise = 0.1

# Prior state
state0 = ([0.], reshape([1e-12], 1,1))

"Generate data"

# Generate signal
observations, states = LGDS(transition_coeffs,
                            emission_coeffs,
                            process_noise,
                            measurement_noise,
                            state0;
                            time_horizon=T)

# Check signal visually
plot(1:T, states[2:end], linewidth=3, color="red", label="states")
scatter!(1:T, observations[1:end], color="blue", label="observations")


"""Basic Kalman smoother"""

# Call filter
msk, Psk = rts_smoother(observations,
                        transition_coeffs,
                        emission_coeffs,
                        process_noise,
                        measurement_noise,
                        state0)

# Visualize estimates
scatter(1:T, observations[1,:], color="black", label="observations")
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, msk[:], color="purple", label="inferred")
plot!(1:T, msk[:],
      ribbon=[sqrt.(Psk[1,1,:]), sqrt.(Psk[1,1,:])],
      color="purple", alpha=0.1, label="")
xlabel!("time [k]")
ylabel!("signal")
savefig("./figures/LGDS_rts.png")
