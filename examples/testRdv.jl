using BenchmarkTools
using DifferentialEquations
using Distributions
using ForwardDiff
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools

# This is a quite interesting and challenging problem with nonlinear measurements and
# discrepancy between filter propagation model (linear RDV CW equations) and true dynamics
# (nonlinear relative dynamics in LVLH). EKF, ESKF, and UKF fail to ensure consistency,
# while IEKF works really well (even for very low number of iterations).
#
# Problem Reference:
# [1] Michaelson, Popov, Zanetti, RECURSIVE UPDATE FILTERING: A NEW APPROACH

function main()
    #Random.seed!(1234)

    # Kalman functions
    n = 0.001131
    μ = 3.986e14
    Rorb = (μ/n^2)^(1/3)
    Δt = 2.0

    # Define Navigation Problem
    Jf(t, x) = [zeros(3, 3) I; [zeros(1, 5) 2n; 0.0 -n^2 zeros(1, 4); 0.0 0.0 3n^2 -2n 0.0 0.0]]
    #f(t, x) = Jf(t, x)*x
    Φ = exp(Jf(0.0, zeros(6)).*Δt)
    rangeLosMeas(x) = [norm(x[1:3]); atan(x[2], x[1]); asin(x[3]/norm(x[1:3]))]
    h(t, x) = (rangeLosMeas(x), diagm([0.1; 0.1π/180; 0.1π/180].^2), ForwardDiff.jacobian(rangeLosMeas, x))  # ỹ, R, H

    # Define Kalman filter
    function kalmanFilter!(nav, Δt, ty, y, Q)
        # Update step at t[k-1] with y[k-1]
        kalmanUpdate!(nav, ty, y, h)
        if hasfield(typeof(nav), :P); nav.P = 0.5(nav.P + nav.P') end       # Sym P

        # Propagate state from t[k-1] to t[k]
        nav.t += Δt
        nav.x .= Φ*nav.x
        nav.P .= Φ*nav.P*Φ' + Q
        #kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps=5)    # [SHOULD USE THIS FOR UKF] Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
    end

    function trueDyn(t, x)
        ω = [0.0; -n; 0.0]
        rT = [0.0; 0.0; -Rorb]
        rC = rT + x[1:3]
        dg = μ*(rT./Rorb^3 - rC./norm(rC)^3)
        return [x[4:6]; -2ω × x[4:6] - ω × (ω × x[1:3]) + dg]
    end

    # Init plot
    set_theme!(theme_fra())
    fig = Figure(size=(1100, 670)); display(fig)
    axs = [
        GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="y [m]", limits=(0, 200, nothing, nothing), title="Nav performance"),
        GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="z [m]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(0, 200, nothing, nothing)),
        GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(0, 200, nothing, nothing)),
        #GLMakie.Axis(fig[3, 1:3]; xlabel="V-bar [m]", ylabel="R-bar [m]", xreversed=true, yreversed=true),
    ]
    function plotnav(ax, T, X, X̂, σ)
        lines!(ax, T, X - X̂; color=:white)
        lines!(ax, T, +3σ; linewidth=2, color=:red)
        lines!(ax, T, -3σ; linewidth=2, color=:red)
    end

    # Run Monte-Carlo
    x₀ = [100; 0.0; 5.0; -0.055; 0.0; -0.085]
    P₀ = diagm([10.0; 10.0; 10.0; 0.05; 0.05; 0.05].^2)
    Q = computeQd(Jf(0.0, zeros(6)), [zeros(3, 3); I], 1e-6I, Δt)

    for nSim in 1:100
        @show nSim
        x̂₀ = x₀ + rand(MvNormal(P₀))
        nav = NavState(0.0, x̂₀, P₀; type=:IEKF, iter=3)
        x = copy(x₀)
        X = [x]; T = [0.0];
        X̂ = [getState(nav)]; σ = [getStd(nav)];

        for k in 1:100
            # Generate measurement at t[k]
            ty = (k - 1)*Δt
            y, R, ~ = h(ty, x)
            y = y + rand(MvNormal(R))

            # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
            kalmanFilter!(nav, Δt, ty, y, Q)

            # Propagate true dynamics from x[k] to x[k+1]
            x = KalmanFilterEngine.odeCore(0, x, Δt, trueDyn; nSteps=5)# + rand(MvNormal(Q))

            # Save data for post-processing
            push!(T, nav.t)
            push!(X, x)
            push!(X̂, getState(nav))
            push!(σ, getStd(nav))
        end

        # Plotting results
        for i in 1:6
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i))
        end
        #lines!(axs[7], getindex.(X, 1), getindex.(X, 3))
    end
    return nothing
end

main();
#@btime main(showplot=false)
#@profview main(showplot=false)
