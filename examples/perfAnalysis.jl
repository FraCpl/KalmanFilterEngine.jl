using BenchmarkTools
using Distributions
using KalmanFilterEngine
using LinearAlgebra

function main()
    P₀ = generatePosDefMatrix(6)
    x₀ = zeros(6)
    Δt = 0.35

    f(t, x) = [x[4:6]; zeros(3)]
    Jf(t, x) = [zeros(3,3) I; zeros(3,6)]
    h(t, x) = (x[1:3], 0.483*Matrix(I,3,3), [I zeros(3,3)])
    Q = computeQd(Jf(0.0,zeros(6)), [zeros(3, 3); I], 0.005616*Matrix(I,3,3), Δt)
    dummy, R, H = h(0,zeros(6))

    nav = NavState(0.0, x₀, P₀)

    y = H*x₀ + rand(MvNormal(R))     # Generate measurement

    @btime kalmanUpdate!(nav, 0.0, y, h)
    @btime kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = 1)
end
main()
