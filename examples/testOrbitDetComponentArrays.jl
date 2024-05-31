using BenchmarkTools
using DifferentialEquations
using Distributions
using ForwardDiff
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools
using ComponentArrays
#=
# Define Navigation Problem - OD1
f(t, x) = [x[4:6]; -3.986e14/norm(x[1:3])^3*x[1:3]]
Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
rangeLosMeas(x) = [norm(x[1:3]); atan(x[2], x[1]); asin(x[3]/norm(x[1:3]))]
h(t, x) = (rangeLosMeas(x), diagm([500; 0.001; 0.001].^2), ForwardDiff.jacobian(rangeLosMeas, x))  # ỹ, R, H
=#

# Define Navigation Problem - OD2
f(t, x) = [x.vel; zeros(3)]
Jf(t, x) = [zeros(3, 3) I; zeros(3, 6)]
h(t, x) = (x[1:3], diagm([10.0; 10.0; 10.0].^2), [I zeros(3, 3)])  # ỹ, R, H

# Define Kalman filter
function kalmanFilter!(nav, Δt, ty, y, Q)
    kalmanUpdate!(nav, ty, y, h)                                        # Update step at t[k-1] with y[k-1]
    if hasfield(typeof(nav), :P); nav.P = 0.5(nav.P + nav.P') end       # Sym P
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = ceil(Int, Δt/10.0))    # Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
end

# Run
function main(;showplot=true)
    #Random.seed!(1234)

    x̂₀ = ComponentArray(
        pos=[6370e3+500e3; 0.0; 0.0],
        vel=[0.0; 1.1*sqrt(3.986e14/(6370e3+500e3)); 532.2],
    )
    P₀ = x̂₀*x̂₀'
    P₀ .= Diagonal([1.0e2; 1.0e2; 1.0e2; 1.0e2; 1.0e2; 1.0e2].^2)

    nav = NavState(0.0, x̂₀, P₀)
    navUD = NavState(0.0, x̂₀, P₀; type=:UD)
    navUKF = NavState(0.0, x̂₀, P₀; type=:UKF)
    navSRUKF = NavState(0.0, x̂₀, P₀; type=:SRUKF)
    navIEKF = NavState(0.0, x̂₀, P₀; type=:IEKF)

    Δt = 100.0
    Q = computeQd([zeros(3,3) I; zeros(3, 6)], [zeros(3, 3); I], 0.01I, Δt)

    x = nav.x + rand(MvNormal(getCov(nav)))
    X = [x]; T = [0.0];
    X̂ = [nav.x]; σ = [getStd(nav)];
    X̂ud = [navUD.x]; σud = [getStd(navUD)];
    X̂ukf = [navUKF.x]; σukf = [getStd(navUKF)];
    X̂srukf = [navSRUKF.x]; σsrukf = [getStd(navSRUKF)]
    X̂iekf = [navIEKF.x]; σiekf = [getStd(navIEKF)]

    for k in 1:100
        # Generate measurement at t[k]
        ty = (k - 1)*Δt
        y, R, ~ = h(ty, x)
        y = y + rand(MvNormal(R))

        # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
        kalmanFilter!(nav, Δt, ty, y, Q)
        kalmanFilter!(navUD, Δt, ty, y, Q)
        kalmanFilter!(navUKF, Δt, ty, y, Q)
        kalmanFilter!(navSRUKF, Δt, ty, y, Q)
        kalmanFilter!(navIEKF, Δt, ty, y, Q)

        # Propagate true dynamics from x[k] to x[k+1]
        x = KalmanFilterEngine.odeCore(0, x, Δt, f; nSteps=1) + rand(MvNormal(Q))

        # Save data for post-processing
        #if showplot
        push!(T, nav.t)
        push!(X, x)
        push!(X̂, nav.x)
        push!(σ, getStd(nav))
        push!(X̂ud, navUD.x)
        push!(σud, getStd(navUD))
        push!(X̂ukf, navUKF.x)
        push!(σukf, getStd(navUKF))
        push!(X̂srukf, navSRUKF.x)
        push!(σsrukf, getStd(navSRUKF))
        push!(X̂iekf, navIEKF.x)
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
        fig = Figure(size=(1100, 670))
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
