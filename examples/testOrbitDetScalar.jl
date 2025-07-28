using BenchmarkTools
using DifferentialEquations
using Distributions
# using ForwardDiff
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools

#=
# Define Navigation Problem - OD1
f(t, x) = [x[4:6]; -3.986e14/norm(x[1:3])^3*x[1:3]]
Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
rangeLosMeas(x) = [norm(x[1:3]); atan(x[2], x[1]); asin(x[3]/norm(x[1:3]))]
h(t, x) = (rangeLosMeas(x), diagm([500; 0.001; 0.001].^2), ForwardDiff.jacobian(rangeLosMeas, x))  # ỹ, R, H
=#

# Define Navigation Problem - OD2
f(t, x) = [x[4:6]; zeros(3); zeros(3)]
Jf(t, x) = [zeros(3, 3) I zeros(3, 3); zeros(6, 9)]
h(t, x) = (x[1:3] + x[7:9], diagm([10.0; 10.0; 10.0].^2), [I zeros(3, 3) I])  # ỹ, R, H

# Define Kalman filter
function kalmanFilter!(nav, Δt, ty, y, Q, isScalar)
    if isScalar
        kalmanUpdateErrorScalar!(nav, ty, y, h)
    else
        kalmanUpdateError!(nav, ty, y, h)                                        # Update step at t[k-1] with y[k-1]
    end
    nav.x .+= nav.δx
    nav.δx .= 0.0
    if hasfield(typeof(nav), :P); nav.P = 0.5(nav.P + nav.P') end       # Sym P
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = ceil(Int, Δt/10.0))    # Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
end

# Run
function main(;showplot=true)
    Random.seed!(1234)

    x̂₀ = [6370e3+500e3; 0.0; 0.0; 0.0; 1.1*sqrt(3.986e14/(6370e3+500e3)); 532.2; zeros(3)]
    P₀ = diagm([1.0e3; 1.0e3; 1.0e3; 1.0e2; 1.0e2; 1.0e2; 0.7; 0.7; 0.7].^2)

    nav = NavState(0.0, x̂₀, P₀, 6)
    navS = NavState(0.0, x̂₀, P₀, 6)

    Δt = 100.0
    Q = computeQd([zeros(3, 3) I zeros(3, 3); zeros(6, 9)], [zeros(3, 3); I; zeros(3, 3)], 0.01I, Δt)

    x = nav.x + rand(MvNormal(getCov(nav)))
    X = [x]; T = [0.0]
    X̂ = [getState(nav)]; σ = [getStd(nav)]
    X̂s = [getState(navS)]; σs = [getStd(navS)]

    for k in 1:100
        # Generate measurement at t[k]
        ty = (k - 1)*Δt
        y, R, ~ = h(ty, x)
        y .+= rand(MvNormal(R))

        # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
        kalmanFilter!(nav, Δt, ty, y, Q, false)
        kalmanFilter!(navS, Δt, ty, y, Q, true)

        # Propagate true dynamics from x[k] to x[k+1]
        #sol = solve(ODEProblem((x, p, t) -> f(t, x), x, (0, Δt)))
        x = KalmanFilterEngine.odeCore(0, x, Δt, f; nSteps=1) + [rand(MvNormal(Q[1:6, 1:6])); zeros(3)]

        # Save data for post-processing
        #if showplot
        push!(T, nav.t)
        push!(X, x)
        push!(X̂, getState(nav))
        push!(X̂s, getState(navS))
        push!(σ, getStd(nav))
        push!(σs, getStd(navS))
        #end
    end

    # Plotting results
    if showplot
        function plotnav(ax, T, X, X̂, σ; kwargs...)
            lines!(ax, T, X - X̂; kwargs...)
            lines!(ax, T, +3σ; linewidth=2, kwargs...)
            lines!(ax, T, -3σ; linewidth=2, kwargs...)
        end
        set_theme!(theme_fra())
        fig = Figure(size=(1100, 670))
        axs = [
                GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="y [m]", title="Nav performance", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="z [m]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[3, 1]; xlabel="Time [s]", ylabel="bx [m]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[3, 2]; xlabel="Time [s]", ylabel="by [m]", limits=(T[1], T[end], nothing, nothing)),
                GLMakie.Axis(fig[3, 3]; xlabel="Time [s]", ylabel="bz [m]", limits=(T[1], T[end], nothing, nothing)),
            ]
        for i in 1:9
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i); color=:white)
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂s, i), getindex.(σs, i); color=:red)
        end
        display(fig)
    end
    return nothing
end

main();
#@btime main(showplot=false)
#@profview main(showplot=false)
