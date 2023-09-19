using LinearAlgebra
using KalmanFilterEngine
using Quaternions
using DifferentialEquations
using Distributions
using ForwardDiff
using Plots#: plot!, plot, plotlyjs
using KeplerianOrbits: KepOrbit, getOrbitState
plotlyjs()
#plotly()

# Global constants
const μ = 4.89
const ω = 4.070264113792746e-04   # planet rotation rate along zB ≡ zI
planetAttitude(t) = [cos(ω*t/2.0); 0.0; 0.0; sin(ω*t/2.0)]  # q_IP

# Define Navigation Problem
f(t, x) = [x[4:6]; -μ/norm(x[1:3])^3*x[1:3]; zeros(3)]
Jf(t, x) = ForwardDiff.jacobian(x -> f(t,x), x)

function voMeas(x, q1_IB, t1, t2)
    # x1 ≡ x[k-1], x2 ≡ x[k]
    q1_IP = planetAttitude(t1)
    q1_PI = q_transpose(q1_IP)
    q2_PI = q_transpose(planetAttitude(t2))
    pos1_P = q_transformVector(q1_PI, x[7:9])
    pos2_P = q_transformVector(q2_PI, x[1:3])
    q1_BP = q_multiply(q_transpose(q1_IB), q1_IP)
    return q_transformVector(q1_BP, normalize(pos2_P - pos1_P))
end
h(x, q1_IB, t1, t2) = (
    voMeas(x, q1_IB, t1, t2),
    Matrix((0.01^2)*I,3,3),
    ForwardDiff.jacobian(x -> voMeas(x, q1_IB, t1, t2), x))  # ỹ, R, H

# Define Kalman filter
function kalmanFilter!(nav, Δt, ty, y, Q, q1_IB)

    # Update step at t[k-1] with y[k-2,k-1]
    if nav.t > 0
        kalmanUpdate!(nav, ty, y, (t, x) -> h(x, q1_IB, nav.t - Δt, nav.t))
    end

    if hasfield(typeof(nav),:P)
        nav.P = 0.5(nav.P + transpose(nav.P))
    end

    # Latch state at t[k-1]
    nav.x̂[7:9] = nav.x̂[1:3]
    H = [I zeros(6,3); I zeros(3,6)]
    nav.P = H*nav.P*transpose(H)

    # Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = ceil(Int,Δt/10.0))
end

# Spacecraft attitude pointing
function spacecraftAttitude(x_I)
    zB_I = normalize(-x_I[1:3])
    yB_I = normalize(x_I[4:6] × x_I[1:3])
    xB_I = normalize(yB_I × zB_I)
    return q_fromAxes(xB_I, yB_I, zB_I) # q_IB
end

# Run
function main()

    # Time parameters
    Δt = 60.0
    Tof = 2*24*3600.0
    t = 0:Δt:Tof

    # Initial State
    x₀ = getOrbitState(KepOrbit(μ = μ, a = 2e3, e = 1.5, θ = -115.0*π/180))

    # Initialize navigation
    #Q = computeQd([zeros(3,3) I zeros(3,3); zeros(6,9)], [zeros(3,3); I; zeros(3,3)], 1e-7I, Δt)
    Q = Diagonal([1e-2*ones(3); 1e-5*ones(3); zeros(3)].^2)  # Polimi
    P₀ = Diagonal([ones(3); 1e-4*ones(3); 1e-6*ones(3)].^2)
    x̂₀ = [x₀; 0.0; 0.0; 0.0] + rand(MvNormal(P₀)); x̂₀[7:9] .= 0.0
    nav = NavState(0.0, x̂₀, P₀)

    # Run navigation
    x = x₀
    X = [x]; T = [t[1]]
    X̂ = [nav.x̂]; σ = [getStd(nav)];
    qOld_IP = zeros(4)
    qOld_IB = zeros(4)
    posOld_P = zeros(3)
    dummy, R, ~ = h(zeros(9), zeros(4), 0.0, 0.0)

    for k = 1:lastindex(t)

        # Generate measurement at t[k]
        q_IP = planetAttitude(t[k])
        q_IB = spacecraftAttitude(x)
        pos_P = q_transformVector(q_transpose(q_IP), x[1:3])
        qOld_BP = q_multiply(q_transpose(qOld_IB), qOld_IP)    # t[k-1]
        y = q_transformVector(qOld_BP, normalize(pos_P - posOld_P)) + rand(MvNormal(R))

        # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
        # NB: First measurement at k == 1 (t = 0) is skipped as it is meaningless
        kalmanFilter!(nav, Δt, 0.0, y, Q, qOld_IB)

        # Propagate true dynamics from x[k] to x[k+1]
        sol = solve(ODEProblem((x, p, t) -> f(t, x)[1:6], x, (0, Δt)))
        x = sol.u[end] + rand(MvNormal(Q[1:6,1:6]))

        # Update "old" variables to be able to generate VO measurements
        # at the next simulation cycle
        qOld_IP = q_IP
        qOld_IB = q_IB
        posOld_P = pos_P

        # Save data for post-processing
        push!(T, T[end] + Δt)
        push!(X, x)
        push!(X̂, nav.x̂)
        push!(σ, getStd(nav))
    end

    function plotnav(i, T, X, X̂, σ, linestyle)
        plot!(T, X - X̂; ticks=:native, lab="", linestyle=linestyle, subplot = i)
        plot!(T, +3σ; color=:red, lab="", linestyle=linestyle, subplot = i)
        plot!(T, -3σ; color=:red, lab="", linestyle=linestyle, subplot = i)
    end

    pp = plot(layout = (2,3))
    lbl = ["x [m]"; "y [m]"; "z [m]"; "vx [m/s]"; "vy [m/s]"; "vz [m/s]"]
    for i in 1:6
        plotnav(i,T/3600.0/24.0,getindex.(X,i),getindex.(X̂,i),getindex.(σ,i),:solid)
        plot!(subplot =i, top_margin = 5*Plots.mm, bottom_margin = 5*Plots.mm, left_margin = 0*Plots.mm, right_margin = 0*Plots.mm)
        xlabel!(subplot = i, "Time [days]"); ylabel!(subplot = i, lbl[i])
        if i == 2; title!(subplot = i, "Nav performance"); end
    end
    display(plot(pp, size = (1100,670), bg = RGB(40/255, 44/255, 52/255), fg = RGB(0.7,0.7,0.7), right_margin = 10*Plots.mm))

    # display(plot(t/3600.0/24.0,norm.(X); ticks = :native))
end

main();
