using LinearAlgebra
using KalmanFilterEngine
using Quaternions
using DifferentialEquations
using Distributions
using ForwardDiff
using Plots#: plot!, plot, plotlyjs
#using OrbitalMechanics: KepOrbit, getState
#plotlyjs()
#plotly()

# Global constants
const μ = 4.89
const ω = 4.070264113792746e-04   # planet rotation rate along zB ≡ zI
planetAttitude(t) = [cos(ω*t/2.0); 0.0; 0.0; sin(ω*t/2.0)]  # q_IP

# Define Navigation Problem
f(t, x) = [x[4:6]; -μ/norm(x[1:3])^3*x[1:3]; zeros(6)]
Jf(t, x) = ForwardDiff.jacobian(x -> f(t,x), x)

function voMeas(x, q1_IB, q2_IB, t1, t2, t3)
    # x1 = x[k-2], x2 ≡ x[k-1], x3 ≡ x[k]
    q1_IP = planetAttitude(t1)
    q1_PI = q_transpose(q1_IP)
    q2_IP = planetAttitude(t2)
    q2_PI = q_transpose(q2_IP)
    q3_PI = q_transpose(planetAttitude(t3))
    pos1_P = q_transformVector(q1_PI, x[7:9])       # x[k-2]
    pos2_P = q_transformVector(q2_PI, x[10:12])     # x[k-1]
    pos3_P = q_transformVector(q3_PI, x[1:3])       # x[k]
    q1_BP = q_multiply(q_transpose(q1_IB), q1_IP)
    q2_BP = q_multiply(q_transpose(q2_IB), q2_IP)
    r12_P = pos2_P - pos1_P
    ρ = norm(r12_P)
    r23_P = pos3_P - pos2_P
    return [q_transformVector(q1_BP, r12_P./ρ); q_transformVector(q2_BP, r23_P./ρ)]
end
h(x, q1_IB, q2_IB, t1, t2, t3) = (
    voMeas(x, q1_IB, q2_IB, t1, t2, t3),
    Matrix((0.01^2)*I,6,6),
    ForwardDiff.jacobian(x -> voMeas(x, q1_IB, q2_IB, t1, t2, t3), x))  # ỹ, R, H

# Define Kalman filter
function kalmanFilter!(nav, Δt, ty, y, Q, q1_IB, q2_IB, latch)

    # Update step at t[k-1]
    if latch == 3
        kalmanUpdate!(nav, ty, y, (t, x) -> h(x, q1_IB, q2_IB, nav.t - 2Δt, nav.t - Δt, nav.t))
    end

    if hasfield(typeof(nav),:P)
        nav.P = 0.5(nav.P + transpose(nav.P))
    end

    # Latch state at t[k-2]
    if latch == 1
        nav.x[7:9] = nav.x[1:3]
        H = [I zeros(6,6); I zeros(3,9); zeros(3,12)]
        nav.P = H*nav.P*transpose(H)
    end

    # Latch state at t[k-1]
    if latch == 2
        nav.x[10:12] = nav.x[1:3]
        H = [I zeros(9,3); I zeros(3,9)]
        nav.P = H*nav.P*transpose(H)
    end

    # Propagation step, from t[k-1] to t[k] = t[k-1] + Δt
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = ceil(Int, Δt/10.0))
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
    #x₀ = getState(KepOrbit(μ = μ, a = 2e3, e = 1.5, θ = -115.0*π/180))
    x₀ = [-905.966704421432,-3049.30785605947,1090.04517280359,0.0370191439000898,0.00971259046629204,-0.0120961513397154] # Polimi

    # Initialize navigation
    #Q = computeQd([zeros(3,3) I zeros(3,3); zeros(6,9)], [zeros(3,3); I; zeros(3,3)], 1e-7I, Δt)
    Q = Diagonal([1e-2*ones(3); 1e-5*ones(3); zeros(6)].^2)  # Polimi
    P₀ = Diagonal([100*ones(3); 1e-4*ones(3); 1e-6*ones(6)].^2)
    x̂₀ = [x₀; zeros(6)] + rand(MvNormal(P₀)); x̂₀[7:12] .= 0.0
    nav = NavState(0.0, x̂₀, P₀)

    # Run navigation
    x = [x₀; x₀; x₀]
    X = [x[1:6]]; T = [t[1]]
    X̂ = [nav.x]; σ = [getStd(nav)];
    dummy, R, ~ = h(zeros(12), zeros(4), zeros(4), 0.0, 0.0, 0.0)

    latch = 1

    for k = 1:lastindex(t)

        # Generate measurement at t[k]
        if latch == 3
            qOldOld_IB = spacecraftAttitude(x[13:18])
            qOld_IB = spacecraftAttitude(x[7:12])
            y = voMeas([x[1:9]; x[13:15]], qOldOld_IB, qOld_IB, t[k] - 2Δt, t[k] - Δt, t[k]) + rand(MvNormal(R))
        else
            qOldOld_IB = zeros(4)
            qOld_IB = zeros(4)
            y = zeros(6)
        end

        # Perform Kalman Filter step, i.e., update x̂[k] and propagate to x̂[k+1]
        # NB: First measurement at k == 1 (t = 0) is skipped as it is meaningless
        kalmanFilter!(nav, Δt, 0.0, y, Q, qOldOld_IB, qOld_IB, latch)
        latch += 1
        if latch > 3
            latch = 1
        end

        # Propagate true dynamics from x[k] to x[k+1]
        x[7:12] = x[13:18]
        x[13:18] = x[1:6]
        sol = solve(ODEProblem((x, p, t) -> f(t, x)[1:6], x[1:6], (0, Δt)))
        x[1:6] .= sol.u[end] + rand(MvNormal(Q[1:6,1:6]))

        # Save data for post-processing
        push!(T, T[end] + Δt)
        push!(X, x[1:6])
        push!(X̂, nav.x)
        push!(σ, getStd(nav))
    end

    function plotnav(i, T, X, X̂, σ, linestyle)
        Plots.plot!(T, X - X̂; ticks=:native, lab="", linestyle=linestyle, subplot=i)
        Plots.plot!(T, +3σ; color=:red, lab="", linestyle=linestyle, subplot=i)
        Plots.plot!(T, -3σ; color=:red, lab="", linestyle=linestyle, subplot=i)
    end

    pp = Plots.plot(layout=(2, 3))
    lbl = ["x [m]"; "y [m]"; "z [m]"; "vx [m/s]"; "vy [m/s]"; "vz [m/s]"]
    for i in 1:6
        plotnav(i, T/3600.0/24.0, getindex.(X,i), getindex.(X̂,i), getindex.(σ,i), :solid)
        Plots.plot!(subplot=i, margin=5*Plots.mm, xlim=(T[1]/3600.0/24.0, T[end]/3600.0/24.0))
        Plots.xlabel!(subplot=i, "Time [days]"); Plots.ylabel!(subplot=i, lbl[i])
        if i == 2; Plots.title!(subplot=i, "Nav performance"); end
    end
    display(Plots.plot(pp, size=(1100, 670), bg=RGB(40/255, 44/255, 52/255), fg=RGB(0.7, 0.7, 0.7), right_margin=10*Plots.mm))

    # display(plot(t/3600.0/24.0,norm.(X); ticks = :native))
end

main();
