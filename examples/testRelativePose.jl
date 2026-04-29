using BenchmarkTools
using DifferentialEquations
using Distributions
using KalmanFilterEngine
using LinearAlgebra
using GLMakie
using Random
using JTools
using Quats

function multJ(X)
    x, y, z = X
    return [x y z 0 0 0; 0 x 0 y z 0; 0 0 x 0 y z]
end

function updateQuat(q, δθ, qTmp=zeros(4))
    c = 1 / sqrt(4 + δθ' * δθ)
    qTmp[1] = 2 * c
    @inbounds for i in 2:4
        qTmp[i] = c * δθ[i - 1]
    end
    return q_multiply(q, qTmp)
end

function updateNavState!(x, δx)
    for ix in 1:6
        x[ix] += δx[ix]
    end
    x[7:10] .= updateQuat(x[7:10], δx[7:9])
    for ix in 11:lastindex(x)
        x[ix] += δx[ix-1]
    end
    δx .= 0.0
    return x
end


@warn "Work in progress"
function main()

    n = 0.001131
    μ = 3.986e14
    rT = (μ / n^2)^(1/3)
    Δt = 1.0

    target = (
        n=n,                    # [rad/s] Orbital rate
        rT=(μ/n^2)^(1/3),       # [m] Orbital radius
        )

    chaser = (
        R_SC=1.0I(3),           # Sensor accommodation
        posCS_C=zeros(3),       # Sensor accommodation
    )

    # True dynamics
    function dyn(t, X)
        dx = zeros(22)

        # Extract states from state vector
        x, y, z = X[1:3]            # posTC_L
        vx, vy, vz = X[4:6]         # velTC_L
        q_IT = X[7:10]
        ωIT_T = X[11:13]
        jxx, jxy, jxz, jyy, jyz, jzz = X[14:19]

        # Translational non-linear relative dynamics
        n = target.n
        zC = z - target.rT
        rC = sqrt(x * x + y * y + zC * zC)
        irC3 = 1 / rC^3

        dx[1] = vx; dx[2] = vy; dx[3] = vz
        dx[4] = 2*n*vz + (n^2)*x - μ * x * irC3
        dx[5] = -μ * y * irC3
        dx[6] = -2*n*vx + (n^2)*z - μ *(1 / target.rT^2 + zC * irC3)

        # dx[4] = 2*n*vz
        # dx[5] = - n^2 * y
        # dx[6] = -2*n*vx + 3(n^2)*z

        # Target absolute rotational dynamics
        JT_T = [jxx jxy jxz; jxy jyy jyz; jxz jyz jzz]
        dx[7:10] .= q_derivative(q_IT, ωIT_T)
        dx[11:13] = JT_T \ ((JT_T * ωIT_T) × ωIT_T)
        return dx
    end

    function dynJacobian(t, X, qNoise=false)
        n = target.n
        J = zeros(21, 21)
        J[1, 4] = 1.0; J[2, 5] = 1.0; J[3, 6] = 1.0
        J[4, 6] = 2*n
        J[5, 2] = -n^2
        J[6, 3] = 3*n^2
        J[6, 4] = -2*n

        ωIT_T = X[11:13]
        J[7:9, 7:9] = -crossMat(ωIT_T)
        J[7, 10] = 1.0; J[8, 11] = 1.0; J[9, 12] = 1.0

        if !qNoise
            jxx, jxy, jxz, jyy, jyz, jzz = X[14:19]
            JT_T = [jxx jxy jxz; jxy jyy jyz; jxz jyz jzz]
            αIT_T = JT_T \ ((JT_T * ωIT_T) × ωIT_T)

            J[10:12, 10:12] = JT_T \ (crossMat(JT_T * ωIT_T) - crossMat(ωIT_T) * JT_T)
            J[10:12, 13:18] = -JT_T \ (multJ(αIT_T) + crossMat(ωIT_T) * multJ(ωIT_T))
        end
        return J
    end

    Fw = [zeros(3, 6); 1e-3I(3) zeros(3, 3); zeros(3, 6); zeros(3, 3) 1e-4I(3); zeros(9, 6)]
    Q = computeQd(dynJacobian(0.0, zeros(22), true), Fw, I, Δt)

    # xEst = [posTC_L; velTC_L; q_IT; ωIT_T;  JT_T; posTQ_Q]
    # error =[    1:3;     4:6;  7:9; 10:12; 13:18;  19:21]
    # full = [    1:3;     4:6; 7:10; 11:13; 14:19;  20:22]
    R = diagm([0.1π/180; 0.1π/180] .^ 2)
    function featMeas(X, posQF_Q, R_CI, R_IL)
        # Compute measurement
        posTC_L = X[1:3]
        q_IT = X[7:10]
        posTQ_Q = X[20:22]

        R_SC = chaser.R_SC
        posTF_Q = posTQ_Q + posQF_Q
        R_ST = R_SC * R_CI * q_toDcm(q_IT)
        R_SL = R_SC * R_CI * R_IL

        xSF_S, ySF_S, zSF_S = R_ST * posTF_Q - R_SC * chaser.posCS_C - R_SL * posTC_L
        y = [xSF_S / zSF_S; ySF_S / zSF_S]

        # Compute jacobian
        Hlos = [1/zSF_S 0 -xSF_S/zSF_S^2; 0 1/zSF_S -ySF_S/zSF_S^2]
        H = zeros(2, 21)
        H[:, 1:3] = - Hlos * R_SL
        H[:, 7:9] = - Hlos * R_ST * crossMat(posTF_Q)
        H[:, 19:21] = Hlos * R_ST

        return y, R, H
    end

    # Define Kalman filter
    function kalmanFilter!(nav, Δt, y, Q, R_CI, R_IL)

        # Update step at t[k-1] with y[k-1]
        for yk in y
            h(t, x) = featMeas(x, yk.posQF_Q, R_CI, R_IL)
            kalmanUpdateErrorScalar!(nav, 0.0, yk.yMeas, h)
        end

        # Update full state
        updateNavState!(nav.x, nav.δx)
        nav.P .= 0.5 .* (nav.P + nav.P')

        # Propagate state from t[k-1] to t[k]
        # Q = computeQd(dynJacobian(0.0, nav.x, true), Fw, I, Δt)
        kalmanPropagate!(nav, Δt, dyn, dynJacobian, Q; nSteps=5)
    end

    # Init plot
    set_theme!(theme_fra())
    fig = Figure(; size=(1100, 670));
    display(fig)
    axs = (
        GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(0, nothing, nothing, nothing), title="Position estimation performance"),
        GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="y [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 1]; xlabel="Time [s]", ylabel="z [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(0, nothing, nothing, nothing), title="Velocity estimation performance"),
        GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 2]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="θx [deg]", limits=(0, nothing, nothing, nothing), title="Attitude estimation performance"),
        GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="θy [deg]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 3]; xlabel="Time [s]", ylabel="θz [deg]", limits=(0, nothing, nothing, nothing)),        #GLMakie.Axis(fig[3, 1:3];
    )
    function plotnav(ax, T, X, X̂, σ, α=1.0)
        lines!(ax, T, X - X̂; color=(:white, α))
        lines!(ax, T, +3σ; linewidth=2, color=:red)
        lines!(ax, T, -3σ; linewidth=2, color=:red)
    end

    # Run Monte-Carlo
    x0_L = [50.0; 5.0; 0.0; 0.0; 0.0; -n*50/2]
    q0_IT = normalize([0.95; 0.05; 0.0; 0.1])

    P₀ = diagm([4.5/3*ones(3); 0.03/3*ones(3); 17.19/3*π/180*ones(3); 3.44/3*π/180*ones(3); 20*ones(6); 0.15/3*ones(3)].^2)

    Rdist = MvNormal(R)
    P0dist = MvNormal(P₀)

    ey = [0.0; 1.0; 0.3]

    for nSim in 1:3
        @show nSim
        Rot = dcm_fromEuler(12*π/180*randn(), 12*π/180*randn(), 12*π/180*randn())
        JT_T = Rot * diagm([3000; 2500.0; 1200.0]) * Rot'
        ω0IT_T = 0.3*randn(3)
        jT_T = [JT_T[1, 1]; JT_T[1, 2]; JT_T[1, 3]; JT_T[2, 2]; JT_T[2, 3]; JT_T[3, 3]]
        posTQ_Q = randn(3)
        x₀ = [x0_L; q0_IT; ω0IT_T; jT_T; posTQ_Q]

        δx₀ = rand(P0dist)
        x̂₀ = copy(x₀)
        ns = 12
        nav = NavState(0.0, updateNavState!(x̂₀, δx₀), P₀, ns)
        x = copy(x₀)
        X = [x];
        T = [0.0];
        X̂ = [getState(nav)];
        σ = [getStd(nav)];
        posQF_Q = [1.7*randn(3) for _ in 1:6]

        for k in 1:round(Int, 3600 / Δt)
            # Generate lvlh and chaser body frame
            sθ, cθ = sincos(target.n * (k - 1) * Δt)
            zL_I = [-cθ; -sθ; 0.0]; yL_I = [0.0; 0.0; -1.0]; xL_I = cross(yL_I, zL_I)
            R_IL = q_toDcm(q_fromAxes(xL_I, yL_I, zL_I))# q_roty!(R_IL, target.n * (k - 1) * Δt)
            zS_L = -normalize(x[1:3])
            yS_L = normalize(cross(zS_L, ey))
            xS_L = cross(yS_L, zS_L)
            q_LS = q_fromAxes(xS_L, yS_L, zS_L)
            R_LS = q_toDcm(q_LS)
            R_CI = (R_IL * R_LS * chaser.R_SC)'

            # Generate measurement at t[k]
            y = []
            for pQF_Q in posQF_Q
                yMeas, _, _ = featMeas(x, pQF_Q, R_CI, R_IL)
                yMeas = yMeas + rand(Rdist)
                push!(y, (yMeas=yMeas, posQF_Q=pQF_Q))
            end

            # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
            kalmanFilter!(nav, Δt, y, Q, R_CI, R_IL)

            # Propagate true dynamics from x[k] to x[k+1]
            x = KalmanFilterEngine.odeCore(0, x, Δt, dyn; nSteps=5)# + rand(MvNormal(Q))

            # Save data for post-processing
            push!(T, nav.t)
            push!(X, x)
            push!(X̂, getState(nav))
            push!(σ, getStd(nav))
        end

        # Plotting results
        α = 0.6 + 0.4*rand()
        for i in 1:6
            plotnav(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i), α)
        end

        qEst_IT = getindex.(X̂, [[7, 8, 9, 10]])
        qTrue_IT = getindex.(X, [[7, 8, 9, 10]])
        qErr_IB = q_attitudeError.(qTrue_IT, qEst_IT)
        for i in 7:9
            plotnav(axs[i], T, getindex.(qErr_IB, i-6)*180/π, zero(T), getindex.(σ, i)*180/π, α)
        end

        #lines!(axs[7], getindex.(X, 1), getindex.(X, 3))
    end
    return nothing
end

main();
#@btime main(showplot=false)
#@profview main(showplot=false)
