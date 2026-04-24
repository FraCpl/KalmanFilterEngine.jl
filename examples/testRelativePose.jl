using BenchmarkTools
using DifferentialEquations
using Distributions
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools
using Quats

@warn "Work in progress"
function main()

    #Random.seed!(1234)

    # Kalman functions
    n = 0.001131
    μ = 3.986e14
    rT = (μ / n^2)^(1/3)
    Δt = 2.0

    JT_T = 1.0I(3)
    posTQ_Q = randn(3)
    posQF_Q = [randn(3) for _ in 1:10]
    target = (
        n=n,                    # [rad/s] Orbital rate
        rT=(μ/n^2)^(1/3),       # [m] Orbital radius
        JT_T=JT_T,              # [kg m²] Inertia matrix
        invJT_T = inv(JT_T),    # [kg⁻¹ m⁻²] Inverse inertia matrix
        posTQ_Q=posTQ_Q,        # [m] Position of Q wrt CoM T
        posQF_Q=posQF_Q         # [m] Position of features F in Q
        )

    chaser = (
        R_SC=1.0I(3),           # Sensor accommodation
        posCS_C=zeros(3),       # Sensor accommodation
    )

    # Define Navigation Problem
    Jf = [zeros(3, 3) I; zeros(3, 6)]
    Jf[4, 6] = 2 * n
    Jf[5, 2] = -n^2
    Jf[6, 3] = 3n^2
    Jf[6, 4] = -2 * n
    Φ = exp(Jf .* Δt)

    R = diagm([0.1; 0.1π/180; 0.1π/180] .^ 2)
    yOut = zeros(2)
    H = zeros(3, 12)

    # xEst = [posTC_L; velTC_L; q_IT; ωIT_T; posTQ_Q; JT_T]
    function featMeas(X, posQF_Q, R_CI, R_IL)
        posTC_L = X[1:3]
        q_IT = X[7:10]
        posTQ_Q = X[14:16]

        R_IT = q_toDcm(q_IT)
        R_SC = chaser.R_SC
        posCS_C = chaser.posCS_C

        posSF_S = R_SC*(R_CI * R_IT * (posTQ_Q - posQF_Q) - posCS_C - R_CI * R_IL * posTC_L)
        y = [posSF_S[1]/posSF_S[3]; posSF_S[2]/posSF_S[3]]

# TODO: to be updated
        yOut[1] = sqrt(x * x + y * y + z * z)
        yOut[2] = atan(y, x)
        yOut[3] = asin(z / yOut[1])
        return yOut
    end

    function jacMeas(X)
        x, y, z, ~, ~, ~ = X
        rip2 = x * x + y * y
        r2 = rip2 + z * z
        c2 = 1 / r2 / sqrt(rip2)
        ir = 1 / sqrt(r2)
        H[1, 1] = x * ir
        H[1, 2] = y * ir
        H[1, 3] = z * ir
        H[2, 1] = -y / rip2
        H[2, 2] = x / rip2
        H[3, 1] = -c2 * x * z
        H[3, 2] = -c2 * y * z
        H[3, 3] = c2 * rip2
        return H
    end

    h(t, x) = (featMeas(x), R, jacMeas(x))#ForwardDiff.jacobian(rangeLosMeas, x))  # ỹ, R, H

    # Define Kalman filter
    xTmp = zeros(6)
    function kalmanFilter!(nav, Δt, ty, y, Q)
        # Update step at t[k-1] with y[k-1]
        kalmanUpdateIter!(nav, ty, y, h, 3)
        # nav.P .= 0.5(nav.P + nav.P')

        # Propagate state from t[k-1] to t[k]
        mul!(xTmp, Φ, nav.x)
        nav.x .= xTmp
        nav.t += Δt
        kalmanPropagateCov!(nav, Φ, Q)
    end

    dx = zeros(13)
    function trueDyn(t, X)      # OK
        # Translational non-linear relative dynamics
        n = target.n
        x, y, z = X[1:3]        # posTC_L
        vx, vy, vz = X[4:6]        # velTC_L

        xC = x; yC = y; zC = z - target.rT
        rC = sqrt(xC * xC + yC * yC + zC * zC)
        irC3 = 1 / rC^3

        dx[1] = vx; dx[2] = vy; dx[3] = vz
        dx[4] = 2*n*vz + (n^2)*x - μ * xC * irC3
        dx[5] = -μ * yC * irC3
        dx[6] = -2*n*vx + (n^2)*z - μ *(1 / target.rT^2 + zC * irC3)

        # Target absolute rotational dynamics
        q_IT = X[7:10]
        ωIT_T = X[11:13]

        q_derivative!(dx[7:10], q_IT, ωIT_T)
        dx[11:13] = target.invJT_T * (ωIT_T × (target.JT_T * ωIT_T))
        return dx
    end

    # Init plot
    set_theme!(theme_fra())
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
        lines!(ax, T, X - X̂; color=:white)
        lines!(ax, T, +3σ; linewidth=2, color=:red)
        lines!(ax, T, -3σ; linewidth=2, color=:red)
    end

    # Run Monte-Carlo
    x₀ = [100; 0.0; 5.0; -0.055; 0.0; -0.085]
    P₀ = diagm([10.0; 10.0; 10.0; 0.05; 0.05; 0.05] .^ 2)
    Q = computeQd(Jf, [zeros(3, 3); I], 1e-6I, Δt)
    Rdist = MvNormal(R)
    P0dist = MvNormal(P₀)

    for nSim in 1:100
        @show nSim
        x̂₀ = x₀ + rand(P0dist)
        nav = NavState(0.0, x̂₀, P₀)
        x = copy(x₀)
        X = [x];
        T = [0.0];
        X̂ = [getState(nav)];
        σ = [getStd(nav)];

        for k in 1:100
            # Generate measurement at t[k]
            ty = (k - 1)*Δt
            y, R, ~ = h(ty, x)
            y = y + rand(Rdist)

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
