using Distributions
using KalmanFilterEngine
using LinearAlgebra
using Test

include("testUpdate.jl")

function TEST_UD()
    n = 12
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    ε = U*diagm(D)*U' - P
    return maximum(abs, (ε))
end

function TEST_generatePosDefMatrix()
    n = 19
    P = generatePosDefMatrix(n)
    λ = eigvals(P)
    ε = maximum(abs, (P - P'))
    return (~(all(isreal(λ)) && minimum(λ) > 0.0))*1.0 + ε
end

function TEST_ageeTurnerUpdate()
    n = 7
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    c = abs(randn())
    x = randn(n)
    UDUtrue = (P + c .* x*x')

    Ũ, D̃ = KalmanFilterEngine.ageeTurnerUpdate(U, D, c, x)
    ε = Ũ*diagm(D̃)*Ũ' - UDUtrue
    return maximum(abs, ε)
end

function TEST_ageeTurnerUpdate!()
    n = 7
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    c = abs(randn())
    x = randn(n)
    UDUtrue = (P + c .* x*x')
    xtmp = zero(x)

    KalmanFilterEngine.ageeTurnerUpdate!(U, D, c, x, xtmp)
    ε = U*diagm(D)*U' - UDUtrue
    return maximum(abs, ε)
end

function TEST_carlsonUpdate()
    n = 7
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    H = randn(1, n)
    R = abs(randn())

    K, U, D, α = KalmanFilterEngine.carlsonUpdate(U, D, H[:], R)

    ε1 = maximum(abs, (U*diagm(D)*U' - (P - K*H*P)))
    ε2 = α - ((H * P * H')[1] + R)
    return maximum([ε1; ε2])
end

function TEST_carlsonUpdate!()
    n = 7
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    H = randn(1, n)
    R = abs(randn())
    K = zeros(n)

    K, α = KalmanFilterEngine.carlsonUpdate!(U, D, H[:], R, K)

    ε1 = maximum(abs, U*diagm(D)*U' - (P - K*H*P))
    ε2 = α - ((H * P * H')[1] + R)
    return maximum([ε1; ε2])
end

function TEST_modGramSchmidt()
    n = 11
    nw = 7
    Φ = randn(n, n)
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    Q = abs.(randn(nw))
    Φw = randn(n, nw)

    Ū, D̄ = KalmanFilterEngine.modGramSchmidt(Φ, U, D, Φw, Q)
    ε1 = Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Φw*diagm(Q)*Φw')

    U2, D2 = KalmanFilterEngine.modGramSchmidtReduced(Φ, U, D)
    ε2 = U2*diagm(D2)*U2' - Φ*P*Φ'

    return maximum([maximum(abs, (ε1)); maximum(abs, (ε2))])
end

function TEST_kalmanOde()
    μ = 3.986e14
    x0 = [6380e3+500e3; 0.0; 1.5e2; 0.0; sqrt(μ/(6380e3+500e3))*1.03; 0.0]
    oc = KalmanFilterEngine.ODECache(x0)

    r = norm(x0[1:3])
    rV²μ = r*(norm(x0[4:6])^2)/μ
    sma = r/(2.0 - rV²μ)

    Torb = 2π*sqrt(sma^3/μ)

    function f!(dx, x, μ, t)
        dx[1:3] .= x[4:6]
        dx[4:6] = -μ/norm(x[1:3])^3*x[1:3]
    end
    x = copy(x0)
    KalmanFilterEngine.odeSolve!(x, 0.0, Torb, f!, μ, oc; nSteps=ceil(Int, Torb/1.0))

    return norm(x[1:3] - x0[1:3]) < 1e-3
end

function TEST_UDpropagate1()
    n = 9
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    Φ = randn(n, n)
    nc = n

    # Case 1: full correlation, diagonal Qxx
    Q = diagm(abs.(randn(n)))
    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_kalmanOdeSTM()

    f!(dx, x, p, t) = @inbounds for i in 1:3; dx[i] = x[i+3]; end
    Jf!(Fx, x, p, t) = @inbounds for i in 1:3; Fx[i, i+3] = 1.0; end

    t0 = 3.0
    x0 = [randn(3); randn(3)]
    Δt = 3.760
    oc = KalmanFilterEngine.ODECache(x0)

    x = copy(x0)
    KalmanFilterEngine.odeSolve!(x, t0, Δt, f!, Jf!, nothing, oc; nSteps=1)
    Φ = oc.Φ
    xTrue = [x0[1:3] + x0[4:6]*Δt; x0[4:6]]
    ΦTrue = I + [zeros(3, 3) Δt*I; zeros(3, 6)]

    return max(maximum(abs, (xTrue - x)), maximum(abs, (ΦTrue - Φ)))
end

function TEST_UDpropagate2()
    n = 9
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    Φ = randn(n, n)
    nc = n

    # Case 2: full correlation, full Qxx
    Q = generatePosDefMatrix(n)
    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_UDpropagate2b()
    n = 6
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)
    Δt = 0.1
    Φ = exp([zeros(3, 3) I; zeros(3, 6)] .* Δt)
    nc = n

    # Case 2: full correlation, full Qxx
    Q = computeQd([zeros(3, 3) I; zeros(3, 6)], [zeros(3, 3); I], 0.01I, Δt)
    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_UDpropagate3()
    n = 9
    P = generatePosDefMatrix(n)
    U, D = KalmanFilterEngine.UD(P)

    # Case 3: full correlation, Noiseless
    nc = n
    Φ = randn(n, n)
    Q = zeros(n, n)

    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_UDpropagate4()
    # Case 4: partial correlation
    # Here we need to carefully redefine P to make sure that the non correlated states
    # have cross correlation with themselves equal to zero, i.e., that Pnn is a
    # diagonal matrix. Yet we perform a STM propagation of P to include cross-correlation
    # between correlated and non correlated states in P, i.e., Pcn, so to have a test
    # covariance matrix P = [Pcc Pcn; Pnc Pnn] with Pnn diagonal and Pcn ≠ 0.
    n = 9
    nc = 6
    Φ = [randn(nc, n); zeros(n-nc, nc) diagm(exp.(-abs.(randn(n-nc))))]
    P = [generatePosDefMatrix(nc) zeros(nc, n-nc); zeros(n-nc, nc) diagm(abs.(randn(n-nc)))]
    P = Φ*P*Φ'
    P = 0.5(P + P')
    U, D = KalmanFilterEngine.UD(P)

    Q = [generatePosDefMatrix(nc) zeros(nc, n-nc); zeros(n-nc, nc) diagm(abs.(randn(n-nc)))]
    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_UDpropagate5()
    n = 9
    nc = 6
    Φ = [randn(nc, n); zeros(n-nc, nc) diagm(exp.(-abs.(randn(n-nc))))]
    P = [generatePosDefMatrix(nc) zeros(nc, n-nc); zeros(n-nc, nc) diagm(abs.(randn(n-nc)))]
    P = Φ*P*Φ'
    P = 0.5(P + P')
    U, D = KalmanFilterEngine.UD(P)

    # Case 5: partial correlation with some zero process noise terms in
    # the non-correlated terms
    Q = [
        generatePosDefMatrix(nc) zeros(nc, n-nc);
        zeros(n-nc, nc) diagm(abs.([0.0; randn(n-nc-1)]))
    ]
    Ū, D̄ = KalmanFilterEngine.UDpropagate(U, D, Φ, Q, nc)
    return maximum(abs, (Ū*diagm(D̄)*Ū' - (Φ*P*Φ' + Q)))
end

function TEST_cholupdate(sgn)
    n = 8
    S = cholesky(generatePosDefMatrix(n)).U
    x = 0.3*randn(n)
    V = copy(S);
    y = copy(x)
    KalmanFilterEngine.cholupdate!(V, y, sgn)
    Vtrue = cholesky(S'*S + sgn*x*x').U
    maximum(abs, (Vtrue - V))
end

function TEST_simpleKalman(type::Symbol)
    P₀ = generatePosDefMatrix(6)
    x₀ = zeros(6)
    Δt = 0.35
    x̂₀ = x₀ + rand(MvNormal(P₀))
    Φ = I + [zeros(3, 3) Δt*I; zeros(3, 6)]

    R = 0.483*Matrix(I, 3, 3)
    H = [I zeros(3, 3)]
    f!(dx, x, p, t) = @inbounds for i in 1:3; dx[i] = x[i+3]; end
    Jf!(Fx, x, p, t) = @inbounds for i in 1:3; Fx[i, i+3] = 1.0; end
    h!(meas, x, p, t) = @inbounds for i in 1:3; meas.y[i] = x[i]; end#(x[1:3], 0.483*Matrix(I, 3, 3), [I zeros(3, 3)])

    J0 = zeros(6, 6)
    Jf!(J0, zeros(6), 0.0, 0.0)
    Q = computeQd(J0, [zeros(3, 3); I], 0.005616*Matrix(I, 3, 3), Δt)
    # dummy, R, H = h(0, zeros(6))

    nav = NavState(0.0, x̂₀, P₀; type=type)
    meas = NavMeasurement(6, 3; R=R, H=H)

    function klm!(nav, meas, y)
        kalmanUpdate!(nav, y, h!, meas)
        kalmanPropagate!(nav, Δt, f!, Jf!, 0.0, Q; nSteps=10)
    end

    function klmSimple(x̂, P, y)
        # Update
        K = (P * H') / (H * P * H' + R)
        x̂ = x̂ + K * (y - H * x̂)
        P = (I - K * H) * P

        # Propagation
        x̂ = Φ * x̂
        P = Φ * P * Φ' + Q
        return x̂, P
    end

    ε = -1e8
    x = copy(x₀)
    x̂ = copy(x̂₀)
    P = copy(P₀)
    Rrand = MvNormal(R)
    Qrand = MvNormal(Q)
    for _ in 1:100
        y = H*x + rand(Rrand)     # Generate measurement
        klm!(nav, meas, y)                    # Execute Kalman step
        x̂, P = klmSimple(x̂, P, y)       # Execute Kalman step (simple)
        ε = maximum([ε maximum(abs, (nav.x - x̂)) maximum(abs, (getCov(nav) - P))])    # Error
        x = Φ*x + rand(Qrand)     # Propagate state
    end

    @show ε
    return ε
end

@testset "KalmanFilterEngine.jl" begin
    ERR_TOL = 1e-8
    @test TEST_UD() < ERR_TOL
    @test TEST_generatePosDefMatrix() < ERR_TOL
    @test TEST_ageeTurnerUpdate() < ERR_TOL
    @test TEST_ageeTurnerUpdate!() < ERR_TOL
    @test TEST_carlsonUpdate() < ERR_TOL
    @test TEST_carlsonUpdate!() < ERR_TOL
    @test TEST_modGramSchmidt() < ERR_TOL
    @test TEST_kalmanOde()
    @test TEST_kalmanOdeSTM() < ERR_TOL
    @test TEST_UDpropagate1() < ERR_TOL
    @test TEST_UDpropagate2() < ERR_TOL
    @test TEST_UDpropagate2b() < ERR_TOL
    @test TEST_UDpropagate3() < ERR_TOL
    @test TEST_UDpropagate4() < ERR_TOL
    @test TEST_UDpropagate5() < ERR_TOL
    @test TEST_cholupdate(+1.0) < ERR_TOL
    @test TEST_cholupdate(-1.0) < ERR_TOL
    @test TEST_simpleKalman(:EKF) < ERR_TOL
    # @test TEST_simpleKalman(:UD) < ERR_TOL
    @test TEST_simpleKalman(:UKF) < 10ERR_TOL
    # @test TEST_simpleKalman(:SRUKF) < 100*ERR_TOL # TODO: Not working!
    @test testUpdate(1)
    @test testUpdate(2)
    @test testUpdate(3)
end
