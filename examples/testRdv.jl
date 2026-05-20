using BenchmarkTools
using DifferentialEquations
using Distributions
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random

# This is a quite interesting and challenging problem with nonlinear measurements and
# discrepancy between filter propagation model (linear RDV CW equations) and true dynamics
# (nonlinear relative dynamics in LVLH). EKF, ESKF, and UKF fail to ensure consistency,
# while IEKF works really well (even for very low number of iterations).
#
# Difference in the solution can be shown by setting iter=1 vs. iter=3.
#
# Problem Reference:
# [1] Michaelson, Popov, Zanetti, RECURSIVE UPDATE FILTERING: A NEW APPROACH

function h!(meas, X, p, t)
    Y = meas.y
    H = meas.H
    R = meas.R
    x, y, z, _, _, _ = X
    rip2 = x * x + y * y
    r2 = rip2 + z * z
    r = sqrt(r2)

    # Measurement
    Y[1] = r
    Y[2] = atan(y, x)
    Y[3] = asin(z / r)

    # Measurement Jacobian
    c2 = 1 / r2 / sqrt(rip2)
    ir = 1 / r
    H[1, 1] = x * ir
    H[1, 2] = y * ir
    H[1, 3] = z * ir
    H[2, 1] = -y / rip2
    H[2, 2] = x / rip2
    H[3, 1] = -c2 * x * z
    H[3, 2] = -c2 * y * z
    H[3, 3] = c2 * rip2

    # Measurement covariance
    R[1, 1] = 0.1^2
    R[2, 2] = R[3, 3] = (0.1 * π / 180)^2

    return nothing
end

function f!(dx, x, p, t)
    n, Rorb, μ = p
    x, y, z, vx, vy, vz = x
    xC = x; yC = y; zC = z - Rorb
    rC = sqrt(xC * xC + yC * yC + zC * zC)
    irC3 = 1 / rC^3

    dgx = -μ * xC * irC3
    dgy = -μ * yC * irC3
    dgz = -μ *(1 / Rorb^2 + zC * irC3)
    dx[1] = vx; dx[2] = vy; dx[3] = vz
    dx[4] = 2*n*vz + (n^2)*x + dgx
    dx[5] = dgy
    dx[6] = -2*n*vx + (n^2)*z + dgz
    return nothing
end

function Jf!(Fx, x, p, t)
    n, Rorb, μ = p
    Fx[1, 4] = 1.0
    Fx[2, 5] = 1.0
    Fx[3, 6] = 1.0
    Fx[4, 6] = 2 * n
    Fx[5, 2] = -n^2
    Fx[6, 3] = 3n^2
    Fx[6, 4] = -2 * n
    return nothing
end

function kalmanFilter!(nav, Φ, Δt, meas, y, Q, p)
    # # UPDATE-OPT1: EKF
    # kalmanUpdate!(nav, y, h!, meas, p)  # Update step at t[k-1] with y[k-1]

    # UPDATE-OPT2: IEKF
    kalmanUpdateIter!(nav, y, h!, meas, p; iter=3)  # Update step at t[k-1] with y[k-1]

    # PROP-OPT1: Discrete-time linear propagation
    nav.x .= Φ*nav.x
    kalmanPropagateCov!(nav, Φ, Q)
    nav.t += Δt

    # # PROP-OPT2: Continuous time non-linear propagation
    # kalmanPropagate!(nav, Δt, f!, Jf!, p, Q)        # Propagate state from t[k-1] to t[k]
end

function main()

    # Kalman functions
    n = 0.001131
    μ = 3.986e14
    Rorb = (μ/n^2)^(1/3)
    Δt = 2.0

    # Init plot
    set_theme!()
    fig = Figure(; size=(1100, 670));
    display(fig)
    axs = [
        GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="y [m]", limits=(0, 200, nothing, nothing), title="Nav performance"),
        GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="z [m]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(0, 200, nothing, nothing)),        #GLMakie.Axis(fig[3, 1:3]; xlabel="V-bar [m]", ylabel="R-bar [m]", xreversed=true, yreversed=true),
    ]
    function plotnav(ax, T, X, X̂, σ)
        lines!(ax, T, X - X̂; color=:black)
        lines!(ax, T, +3σ; linewidth=2, color=:red)
        lines!(ax, T, -3σ; linewidth=2, color=:red)
    end

    # Run Monte-Carlo
    p = (n, Rorb, μ)
    x₀ = [100; 0.0; 5.0; -0.055; 0.0; -0.085]
    P₀ = diagm([10.0; 10.0; 10.0; 0.05; 0.05; 0.05] .^ 2)
    J₀ = zeros(6, 6)
    Jf!(J₀, x₀, p, 0.0)
    Q = computeQd(J₀, [zeros(3, 3); I], 1e-6I, Δt)
    Φ = exp(J₀ .* Δt)
    P0dist = MvNormal(P₀)
    oc = KalmanFilterEngine.ODECache(x₀)
    meas = NavMeasurement(6, 3)
    measTrue = NavMeasurement(6, 3)

    for nSim in 1:100
        @show nSim
        x̂₀ = x₀ + rand(P0dist)
        nav = NavState(0.0, x̂₀, P₀)
        x = copy(x₀)
        X = [copy(x)];
        T = [0.0];
        X̂ = [getState(nav)];
        σ = [getStd(nav)];

        for k in 1:100
            # Generate measurement at t[k]
            h!(measTrue, x, p, 0.0)
            y = measTrue.y + rand(MvNormal(measTrue.R))

            # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
            kalmanFilter!(nav, Φ, Δt, meas, y, Q, p)

            # Propagate true dynamics from x[k] to x[k+1]
            KalmanFilterEngine.odeSolve!(x, 0.0, Δt, f!, p, oc; nSteps=5)

            # Save data for post-processing
            push!(T, nav.t)
            push!(X, copy(x))
            push!(X̂, getState(nav))
            push!(σ, getStd(nav))
        end

        # Plotting results
        for i in 1:6
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i))
        end
    end
    return nothing
end

main()
