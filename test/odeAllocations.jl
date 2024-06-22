using KalmanFilterEngine, BenchmarkTools, ComponentArrays
using LinearAlgebra

@views function odetest(t, x, Δt, f, K1, K2, K3, K4; nSteps=1)
    h = Δt/nSteps

    for _ in 1:nSteps
        f(K1, t, x)
        f(K2, t + 1/3*h, x + h*K1/3)
        f(K3, t + 2/3*h, x  + h*(- K1/3 + K2))
        f(K4, t + h, x + h*(K1 - K2 + K3))

        t += h
        x .+= (K1 + 3K2 + 3K3 + K4)/8
    end
end


function main()

    t0 = 0
    x0 = randn(12)
    Φ0 = randn(12, 12)
    Δt = 1.0
    dx = randn(12)
    dJ = randn(12, 12)
    f(t, x) = dx
    Jf(t, x) = dJ
    #@time x, P = KalmanFilterEngine.odeCore(t0, x0, Φ0, Δt, f, Jf; nSteps=1)

    X = ComponentArray(x = x0; Φ = Φ0)
    dX = ComponentArray(x = zeros(12); Φ = zeros(12, 12))

    function ff!(dX, t, X)
        dX.x = dx
        mul!(dX.Φ, dJ, X.Φ)
        #dX.Φ = dJ*X.Φ
    end

    K1 = similar(X); K2 = similar(X); K3 = similar(X); K4 = similar(X)
    odetest(t0, X, Δt, ff!, K1, K2, K3, K4; nSteps=1)

    #@show norm(x - X.x) + norm(P - X.Φ)
    return nothing
end
@btime main()
