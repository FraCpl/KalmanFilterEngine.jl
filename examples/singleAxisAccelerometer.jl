# Accelerometer (and gyro) bias is commonly modeled as an exponentially correlated random
# variable (ECRV), i.e., a first-order Gauss–Markov process, in navigation filters. However,
# the mean of this process is zero, so the model implicitly assumes that the bias tends to
# decay to zero over time. This raises the question of how well a constant (non-zero) bias
# can be estimated.
#
# With this script we observe that:
# - When β (i.e., 1/τ) is very small (e.g., 1e-5–1e-6), the ECRV behaves almost like a
#   random walk over typical mission durations. In this case, the filter can effectively
#   estimate a constant bias, provided it is observable from measurements and that there is
#   no long measurement outage (without measurements the bias would again converge towards
#   zero).
#
# - If β is larger, the bias estimate will tend to decay toward zero unless continuously
#   and frequently supported by measurements. This does not necessarily make the filter
#   inconsistent, but it can introduce bias in the estimate if the true bias is constant.
#
# - If β = 0, the ECRV reduces to a pure random walk. This model can represent a constant
#   bias exactly (in the absence of process noise), but its covariance grows unbounded over
#   time if there are measurement outages, or if the bias is considered or not observable.
#   This may lead to numerical or consistency issues in long-duration missions if not
#   properly constrained by measurements.
#
# - Introducing an additional constant bias state allows separation between:
#       (i) a constant component, and
#       (ii) a time-varying (ECRV) component.
#   This improves modeling fidelity at the cost of increasing the state dimension, but has
#   the disadvantage that the covariance of the constant state, if persistently updated with
#   measurements, goes to zero (potential numerical errors, and stale nav state).
#
# Important note: usually, in real missions, the IMU bias is calibrated, meaning that most
# of its static component is estimated before its operational use. This means that what we
# need to model in the filter is often just the bias variation and calibration error, which
# is more likely to have a random beavior with zero mean value.
using LinearAlgebra
using KalmanFilterEngine
using GLMakie
using Random

# Nav model ============== #
# Dynamics:
# dr/dt = v                     # Position
# dv/dt = ã - b - d - w_a       # Velocity
# dd/dt = -β * d + w_d          # Gyro bias ECRV/random walk when β = 0
# db/dt = 0                     # Gyro constant bias (can be disabled)
# dby/dt = 0                    # Measurement constant bias
# Measurement
# y = r + by + w_y
# ======================== #
function f!(dx, x, p, t)
    ã, β, βy, enableBias = p
    r, v, d, b, by = x
    dx[1] = v
    dx[2] = ã - d - b * enableBias
    dx[3] = -β * d
    dx[5] = -βy * by
end

function Jf!(Fx, x, p, t)
    ã, β, βy, enableBias = p
    Fx[1, 2] = 1.0
    Fx[2, 3] = -1.0
    Fx[2, 4] = -enableBias
    Fx[3, 3] = -β
    Fx[5, 5] = -βy
end

function h!(meas, x, stdMeas, t)
    r, v, d, b, by = x
    meas.y[1] = r + by
    meas.H[1, 1] = 1.0
    meas.H[1, 5] = 1.0
    meas.R[1, 1] = stdMeas^2
end

function kalmanFilter!(nav, ã, β, Q, Δt, meas, y, stdMeas, βy, enableBias)
    kalmanUpdate!(nav, y, h!, meas, stdMeas)
    kalmanPropagate!(nav, Δt, f!, Jf!, (ã, β, βy, enableBias), Q)
end

function main()

    rng = Random.default_rng()

    # Accelerometer parameters
    # ã = a + d + w_a       # Measured acceleration
    # ḋ = w_ba              # Random walk (with initial bias, d[t₀] = ba)
    Ts = 1 / 8
    vrw = 50.0e-6
    arw = 1e-6
    stdBa = 33.3 * 9.81e-6 * (1e4)

    # Measurement noise
    stdMeas = 1.0
    stdBy = 2.0

    # Initial state uncertainty
    stdPos = 10.0
    stdVel = 1.0

    # Nav modeling
    β = 1e-6*0
    enableBias = false  # Enable constant gyro bias state in nav filter
    nSolveFor = 5
    βy = 0.0

    # Init plot
    set_theme!()
    fig = Figure(; size=(1100, 670));
    display(fig)
    axs = [
        GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="r [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="v [m/s]", limits=(0, nothing, nothing, nothing), title="Nav performance"),
        GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="ba [m/s²]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="by [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 1:2]; xlabel="Time [s]", ylabel="ba [m/s²]", limits=(0, nothing, nothing, nothing)),
    ]
    function plotnav(ax, T, X, X̂, σ)
        lines!(ax, T, X - X̂; color=:black)
        lines!(ax, T, +3σ; linewidth=2, color=:red)
        lines!(ax, T, -3σ; linewidth=2, color=:red)
    end

    # Run Monte-Carlo
    P₀ = diagm([stdPos; stdVel; stdBa; stdBa; stdBy] .^ 2)
    Q = diagm([0.0; Ts * vrw^2; Ts * arw^2; 0.0; Ts * (βy > 0) * 1e-9])
    meas = NavMeasurement(size(P₀, 1), 1)
    sqrtTs = sqrt(Ts)
    H = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 enableBias 0; 0 0 0 0 1]  # to get ba = b + d from Nav state

    for _ in 1:1
        da = stdBa * randn(rng)
        by = stdBy * randn(rng)
        x̂₀ = [stdPos*randn(rng); stdVel*randn(rng); zeros(3)]
        nav = NavState(0.0, x̂₀, P₀; ns=nSolveFor)
        X = [[0.0; 0.0; da; by]]
        T = [0.0]
        X̂ = [H * nav.x]
        σ = [sqrt.(diag(H * nav.P * H'))];

        for _ in 1:round(Int, 900 / Ts)
            # Generate measurement at t[k]
            y = stdMeas * randn(rng) + by

            # Generate IMU reading between t[k] and t[k+1]
            # ã = a + d + w_a
            # ḋ = w_ba
            # d[t₀] = ba
            da += arw * sqrtTs * randn(rng)
            w_a = vrw / sqrtTs * randn(rng)
            ã = da + w_a

            # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
            kalmanFilter!(nav, ã, β, Q, Ts, meas, y, stdMeas, βy, enableBias)

            # Save data for post-processing
            push!(T, nav.t)
            push!(X, [0.0; 0.0; da; by])
            push!(X̂, H * nav.x)
            push!(σ, sqrt.(diag(H * nav.P * H')))
        end

        # Plotting results
        for i in 1:4
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i))
        end

        lines!(axs[5], T, getindex.(X, 3); color=:green)
        lines!(axs[5], T, getindex.(X̂, 3); color=:black)
        lines!(axs[5], T, getindex.(X̂, 3) + 3getindex.(σ, 3); color=:red)
        lines!(axs[5], T, getindex.(X̂, 3) - 3getindex.(σ, 3); color=:red)
    end
    return nothing
end

main()
