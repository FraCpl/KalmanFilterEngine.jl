using KalmanFilterEngine, LinearAlgebra, Distributions, Plots

# True state parameters & state transition matrix
x₀ = zeros(6)                                   # True initial state
Δt = 0.1                                        # Measurement time step
Φ = I + [zeros(3,3) Δt*I; zeros(3,6)]           # State transition matrix

# Define navigation problem
Q = Diagonal([1e-4*ones(3); 1e-3*ones(3)].^2)   # Process noise covariance
R = 0.0483*Matrix(I,3,3)                        # Measurement noise covariance
f(t, x) = [x[4:6]; zeros(3)]                    # System dynamics
h(t, x) = (x[1:3], R, [I zeros(3, 3)])          # Measurement equation

# Initialize navigation state
P₀ = generatePosDefMatrix(6)            # Initial state uncertainty covariance
x̂₀ = x₀ + rand(MvNormal(P₀))            # Initial estimated state
nav = NavState(0.0, x̂₀, P₀; type=:SRUKF)

# Simulate Kalman filter
T = []; X = []; X̂ = []; σ = []
x = x₀
for k in 1:100
    # Generate measurement
    y = x[1:3] + rand(MvNormal(R))

    # Execute Kalman filter step
    kalmanUpdate!(nav, 0.0, y, h)
    kalmanPropagate!(nav, Δt, f, Q)

    # Simulate system dynamics
    x = Φ*x + rand(MvNormal(Q))

    # Save for post-processing
    push!(T, nav.t)
    push!(X, x)
    push!(X̂, nav.x)
    push!(σ, getStd(nav))
end

# Plot results for 1st coordinate
Plots.plot(T, getindex.(X,1) - getindex.(X̂,1), lab="Nav error", xlabel="Time [s]")
Plots.plot!(T, +3.0*getindex.(σ,1); color=:red, lab="3σ")
Plots.plot!(T, -3.0*getindex.(σ,1); color=:red, lab="")
