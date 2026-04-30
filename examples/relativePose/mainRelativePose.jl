using BenchmarkTools
using DifferentialEquations
using Distributions
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools
using Quats

include("plotRelativePose.jl")
include("relnavFilter.jl")

function getLVLH(θ_IL)
    sθ, cθ = sincos(θ_IL)
    return [-sθ 0 -cθ; cθ 0 -sθ; 0 -1 0]    # R_IL
end

function getSensorPointing(pos, R_IL, R_SC)
    zS_L = -normalize(pos)
    yS_L = normalize(cross(zS_L, [0.0; 1.0; 0.3]))
    xS_L = cross(yS_L, zS_L)
    R_LS = dcm_fromAxes(xS_L, yS_L, zS_L)
    R_CI = (R_IL * R_LS * R_SC)'
    return R_CI
end

function main(Nsim=1)
    μ = 3.986e14        # [...] Gravitational constant
    n = 0.001131        # [rad/s] Target orbital rate
    Δt = 1.0            # [s] Measurement sampling time

    R_SC = 1.0I(3)                              # Sensor accommodation
    posCS_C = zeros(3)                          # Sensor accommodation
    x0_L = [50.0; 5.0; 0.0; 0.0; 0.0; -n*50/2]  # Initial relative state

    # Init plot
    axs = initRelativePosePlot()

    # Run Monte-Carlo
    measFun = losMeas#posMeas
    navData = NavData(μ, n, Δt, 1e-3, 1e-4, 0.1, 0.1π/180, R_SC, posCS_C)
    trueData = NavData(μ, n, Δt, 1e-3, 1e-4, 0.1, 0.1π/180, R_SC, posCS_C)

    P₀ = diagm([4.5/3*ones(3); 0.03/3*ones(3); 17.19/3*π/180*ones(3); 3.44/3*π/180*ones(3); 20*ones(6); 0.15/3*ones(3)].^2)
    R = measFun(navData, zeros(22), zeros(3), zeros(3, 3), zeros(3, 3))[2]
    Rdist = MvNormal(R)
    P₀dist = MvNormal(P₀)

    trueDyn(t, x) = navDyn(trueData, x)

    for _ in 1:Nsim
        Rot = dcm_fromEuler(12*π/180*randn(), 12*π/180*randn(), 12*π/180*randn())
        JT_T = Rot * diagm([3000; 2500.0; 1200.0]) * Rot'
        ω0IT_T = 1.5*π/180*randn(3)
        q0_IT = q_random()
        posTQ_Q = randn(3)
        x₀ = [x0_L; q0_IT; ω0IT_T; JT_T[1, 1]; JT_T[1, 2]; JT_T[1, 3]; JT_T[2, 2]; JT_T[2, 3]; JT_T[3, 3]; posTQ_Q]

        δx₀ = rand(P₀dist)
        x̂₀ = copy(x₀)
        nav = NavState(0.0, updateNavState!(navData, x̂₀, δx₀), P₀, 12)
        x = copy(x₀)
        X = [x];
        T = [0.0];
        X̂ = [getState(nav)];
        σ = [getStd(nav)];
        posQF_Q = [1.7*randn(3) for _ in 1:6]

        for k in 1:round(Int, 3600 / Δt)
            # Generate lvlh and chaser body frame
            R_IL = getLVLH(n * (k - 1) * Δt)
            R_CI = getSensorPointing(x[1:3], R_IL, R_SC)

            # Generate measurement at t[k]
            y = []
            for pQF_Q in posQF_Q
                yMeas, _, _ = measFun(trueData, x, pQF_Q, R_CI, R_IL)
                yMeas = yMeas + rand(Rdist)
                push!(y, (yMeas=yMeas, posQF_Q=pQF_Q))
            end

            # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
            kalmanFilter!(nav, navData, y, R_CI, R_IL, measFun)

            # Propagate true dynamics from x[k] to x[k+1]
            x = KalmanFilterEngine.odeCore(0, x, Δt, trueDyn; nSteps=5)# + rand(MvNormal(Q))

            # Save data for post-processing
            push!(T, nav.t)
            push!(X, x)
            push!(X̂, getState(nav))
            push!(σ, getStd(nav))
        end

        # Plotting results
        plotRelativePose(axs, T, X, X̂, σ)
    end
    return nothing
end

main();
#@btime main(showplot=false)
#@profview main(showplot=false)
