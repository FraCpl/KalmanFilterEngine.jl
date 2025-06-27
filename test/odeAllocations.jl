using KalmanFilterEngine, BenchmarkTools, ComponentArrays
using LinearAlgebra

@views function odetest(t, x, Δt, f!; K1=similar(x), K2=similar(x), K3=similar(x), K4=similar(x), tmp=similar(x), nSteps=1)
    h = Δt / nSteps
    @inbounds for _ in 1:nSteps
        f!(K1, t, x)

        @. tmp = x + (h/3) * K1
        f!(K2, t + h/3, tmp)

        @. tmp = x + h*(-K1/3 + K2)
        f!(K3, t + 2h/3, tmp)

        @. tmp = x + h*(K1 - K2 + K3)
        f!(K4, t + h, tmp)

        t += h
        @. x += h*(K1 + 3K2 + 3K3 + K4)/8
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

    function ff!(dx, t, x)
        @inbounds @simd for i in eachindex(x)
            dx[i] = -x[i]
        end
    end
    function ff(t, x)
        dx = similar(x)
        ff!(dx, t, x)
        return dx
    end

    @time ff!(dX, t, X)

    K1 = similar(X); K2 = similar(X); K3 = similar(X); K4 = similar(X); K5 = similar(X)

    XTRUE = KalmanFilterEngine.odeCore(t0, X, Δt, ff)
    odetest(t0, X, Δt, ff!, K1, K2, K3, K4, K5)
    @show norm(XTRUE - X)

    @btime KalmanFilterEngine.odeCore($t0, $X, $Δt, $ff)
    @btime odetest($t0, $X, $Δt, $ff!, $K1, $K2, $K3, $K4, $K5)

    #@show norm(x - X.x) + norm(P - X.Φ)
    return nothing
end
main()
