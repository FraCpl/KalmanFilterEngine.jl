using KalmanFilterEngine, LinearAlgebra, Distributions, GLMakie

function f!(dx, x, p, t)
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
end

function Jf!(Fx, x, p, t)
    @inbounds for i in 1:3
        Fx[i, i+3] = 1.0
    end
end

function main()
    # True state parameters & state transition matrix
    x₀ = zeros(6)                                   # True initial state
    Δt = 0.1                                        # Measurement time step
    Φ = I + [zeros(3, 3) Δt*I; zeros(3, 6)]           # State transition matrix

    # Define navigation problem
    Q = diagm([1e-4*ones(3); 1e-3*ones(3)] .^ 2)   # Process noise covariance
    R = 0.0483*Matrix(I, 3, 3)                        # Measurement noise covariance
    # f(t, x) = [x[4:6]; zeros(3)]                    # System dynamics
    # Jf(t, x) = [zeros(3, 3) I; zeros(3, 6)]
    h(t, x) = (x[1:3], R, [I zeros(3, 3)])          # Measurement equation
    Qrnd = MvNormal(Q)
    Rrnd = MvNormal(R)

    # Initialize navigation state
    P₀ = generatePosDefMatrix(6)            # Initial state uncertainty covariance
    x̂₀ = x₀ + rand(MvNormal(P₀))            # Initial estimated state
    nav = NavState(0.0, x̂₀, P₀)

    # Simulate Kalman filter
    T = [];
    X = [];
    X̂ = [];
    σ = []
    x = x₀
    for k in 1:100
        # Generate measurement
        y = x[1:3] + rand(Rrnd)

        # Execute Kalman filter step
        kalmanUpdate!(nav, 0.0, y, h)
        kalmanPropagate!(nav, Δt, f!, Jf!, 0.0, Q)

        # Simulate system dynamics
        x .= Φ*x + rand(Qrnd)

        # Save for post-processing
        push!(T, nav.t)
        push!(X, x)
        push!(X̂, getState(nav))
        push!(σ, getStd(nav))
    end

    # Plot results for 1st coordinate
    fig = Figure();
    display(fig)
    ax = GLMakie.Axis(fig[1, 1]; ylabel="Nav error", xlabel="Time [s]", limits=(T[1], T[end], -1.5, 1.5))
    lines!(ax, T, getindex.(X, 1) - getindex.(X̂, 1))
    lines!(ax, T, +3.0*getindex.(σ, 1); color=:red)
    lines!(ax, T, -3.0*getindex.(σ, 1); color=:red)
end

main()
