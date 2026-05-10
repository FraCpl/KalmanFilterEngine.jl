using BenchmarkTools
using Distributions
using KalmanFilterEngine
using LinearAlgebra

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
    P₀ = generatePosDefMatrix(6)
    x₀ = zeros(6)
    Δt = 0.35

    h(t, x) = (x[1:3], 0.483*Matrix(I, 3, 3), [I zeros(3, 3)])
    J0 = zeros(6, 6)
    Jf!(J0, zeros(6), 0, 0)
    Q = computeQd(J0, [zeros(3, 3); I], 0.005616*Matrix(I, 3, 3), Δt)
    dummy, R, H = h(0, zeros(6))

    nav = NavState(0.0, x₀, P₀)

    y = H*x₀ + rand(MvNormal(R))     # Generate measurement
    p = 0.0

    @btime kalmanUpdate!($nav, 0.0, $y, $h)
    @btime kalmanPropagate!($nav, $Δt, $f!, $Jf!, $p, $Q)
end
main()
