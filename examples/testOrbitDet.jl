using BenchmarkTools
using DifferentialEquations
using Distributions
# using ForwardDiff
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools

# Define Navigation Problem - OD2
f!(dx, x, p, t) = @inbounds for i in 1:3; dx[i] = x[i+3]; end
Jf!(F, x, p, t) = @inbounds for i in 1:3; F[i, i+3] = 1; end
h(t, x) = (x[1:3], diagm([10.0; 10.0; 10.0] .^ 2), [I zeros(3, 3)])  # ỹ, R, H

# Define Kalman filter
function kalmanFilter!(nav, Δt, ty, y, Q, iter=0)
    if iter == 0
        kalmanUpdate!(nav, ty, y, h)                                        # Update step at t[k-1] with y[k-1]
    else
        kalmanUpdateIter!(nav, ty, y, h, iter)
    end                                 # Update step at t[k-1] with y[k-1]

    # hasfield(typeof(nav), :P) && nav.P = 0.5(nav.P + nav.P')       # Sym P
    kalmanPropagate!(nav, Δt, f!, Jf!, 0, Q; nSteps=ceil(Int, Δt/10.0))    # Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
end

# Run
function main(; showplot=true)
    Random.seed!(1234)

    x̂₀ = [6370e3+500e3; 0.0; 0.0; 0.0; 1.1*sqrt(3.986e14/(6370e3+500e3)); 532.2]
    P₀ = diagm([1.0e3; 1.0e3; 1.0e3; 1.0e2; 1.0e2; 1.0e2] .^ 2)

    nav = NavState(0.0, x̂₀, P₀)
    navUD = NavState(0.0, x̂₀, P₀; type=:UD)
    navUKF = NavState(0.0, x̂₀, P₀; type=:UKF)
    navSRUKF = NavState(0.0, x̂₀, P₀; type=:SRUKF)
    navIEKF = NavState(0.0, x̂₀, P₀)

    Δt = 100.0
    Q = computeQd([zeros(3, 3) I; zeros(3, 6)], [zeros(3, 3); I], 0.01I, Δt)
    Qrnd = MvNormal(Q)

    x = nav.x + rand(MvNormal(getCov(nav)))
    X = [copy(x)];
    T = [0.0]
    X̂ = [getState(nav)];
    σ = [getStd(nav)]
    X̂ud = [getState(navUD)];
    σud = [getStd(navUD)]
    X̂ukf = [getState(navUKF)];
    σukf = [getStd(navUKF)]
    X̂srukf = [getState(navSRUKF)];
    σsrukf = [getStd(navSRUKF)]
    X̂iekf = [getState(navIEKF)];
    σiekf = [getStd(navIEKF)]
    oc = KalmanFilterEngine.ODECache(x)

    for k in 1:100
        # Generate measurement at t[k]
        ty = (k - 1)*Δt
        y, R, ~ = h(ty, x)
        y .+= rand(MvNormal(R))

        # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
        kalmanFilter!(nav, Δt, ty, y, Q)
        kalmanFilter!(navUD, Δt, ty, y, Q)
        kalmanFilter!(navUKF, Δt, ty, y, Q)
        kalmanFilter!(navSRUKF, Δt, ty, y, Q)
        kalmanFilter!(navIEKF, Δt, ty, y, Q, 3)

        # Propagate true dynamics from x[k] to x[k+1]
        KalmanFilterEngine.odeSolve!(x, 0.0, Δt, f!, 0, oc; nSteps=1)
        x .+= rand(Qrnd)

        # Save data for post-processing
        #if showplot
        push!(T, nav.t)
        push!(X, copy(x))
        push!(X̂, getState(nav))
        push!(X̂ud, getState(navUD))
        push!(X̂ukf, getState(navUKF))
        push!(X̂srukf, getState(navSRUKF))
        push!(X̂iekf, getState(navIEKF))
        push!(σ, getStd(nav))
        push!(σud, getStd(navUD))
        push!(σukf, getStd(navUKF))
        push!(σsrukf, getStd(navSRUKF))
        push!(σiekf, getStd(navIEKF))
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
        fig = Figure(; size=(1100, 670))
        axs = [
            GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(T[1], T[end], nothing, nothing)),
            GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="y [m]", title="Nav performance", limits=(T[1], T[end], nothing, nothing)),
            GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="z [m]", limits=(T[1], T[end], nothing, nothing)),
            GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(T[1], T[end], nothing, nothing)),
            GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(T[1], T[end], nothing, nothing)),
            GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(T[1], T[end], nothing, nothing)),
        ]
        for i in 1:6
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i); color=:white)
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂ud, i), getindex.(σud, i); color=:red)
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂ukf, i), getindex.(σukf, i); color=:green)
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂srukf, i), getindex.(σsrukf, i); color=:orange)
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂iekf, i), getindex.(σiekf, i); color=:magenta)
        end
        display(fig)
    end
    return nothing
end

main();
#@btime main(showplot=false)
#@profview main(showplot=false)
