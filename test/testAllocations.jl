using BenchmarkTools
using Distributions
using KalmanFilterEngine
using LinearAlgebra

function testKalmanAllocs()
    f!(dx, x, p, t) = @inbounds for i in 1:3; dx[i] = x[i+3]; end
    Jf!(Fx, x, p, t) = @inbounds for i in 1:3; Fx[i, i+3] = 1.0; end

    P₀ = generatePosDefMatrix(6)
    x₀ = zeros(6)
    Δt = 0.35

    J0 = zeros(6, 6)
    Q = computeQd(J0, [zeros(3, 3); I], 0.005616*Matrix(I, 3, 3), Δt)
    H = [I zeros(3, 3)]

    nav = NavState(0.0, x₀, P₀; type=:EKF)

    R = 0.483*Matrix(I, 3, 3)
    y = H*x₀ + rand(MvNormal(R))     # Generate measurement
    p = nothing
    meas = NavMeasurement(6, 3; H=H, R=R, nReject=1000)
    measScalar = NavMeasurementScalar(6, 3; H=H, R=R, nReject=1000)

    meas.y .= y .+ 1e-6.*randn.()
    measScalar.y .= y .+ 1e-6.*randn.()

    println("kalmanUpdate! (scalar)")
    @btime kalmanUpdate!($nav, $y, $measScalar)
    println("kalmanUpdate!")
    @btime kalmanUpdate!($nav, $y, $meas)
    println("kalmanPropagate!")
    @btime kalmanPropagate!($nav, $Δt, $f!, $Jf!, $p, $Q)
    return nothing
end


function testODEallocs()
    f!(dx, x, p, t) = @inbounds for i in eachindex(dx); dx[i] = randn(); end
    Jf!(F, x, p, t) = @inbounds for i in eachindex(F); F[i] = randn(); end

    t0 = 0
    x0 = randn(12)
    Δt = 1.0
    oc = KalmanFilterEngine.ODECache(x0)
    p = nothing

    println("odeSolve!")
    @btime KalmanFilterEngine.odeSolve!($x0, $t0, $Δt, $f!, $Jf!, $p, $oc)
    return nothing
end


function testUDallocs()
    n = 12
    P = generatePosDefMatrix(12)
    U = zero(P)
    D = zeros(n)
    @btime KalmanFilterEngine.UD!($U, $D, $P)

    KalmanFilterEngine.UD!(U, D, P)
    c = abs(randn())
    x = randn(n)
    xtmp = randn(n)
    @btime KalmanFilterEngine.ageeTurnerUpdate!($U, $D, $c, $x, $xtmp)

    H = randn(n)
    R = abs(randn())
    K = randn(n)
    @btime KalmanFilterEngine.carlsonUpdate!($U, $D, $H, $R, $K)
    return nothing
end

function testSigmaUKFallocs()
    P = generatePosDefMatrix(11)
    x0 = randn(size(P, 1))
    nav = NavState(0.0, x0, P; type=:UKF)
    @btime KalmanFilterEngine.computeSigmaPoints!($nav)
    return nothing
end

function testUKFpropAllocs()
    f!(dx, x, p, t) = @inbounds for i in eachindex(dx); dx[i] = randn(); end
    P = generatePosDefMatrix(11)
    x0 = randn(size(P, 1))
    nav = NavState(0.0, x0, P; type=:UKF)
    Δt = 0.1
    p = nothing
    Q = generatePosDefMatrix(11)
    @btime kalmanPropagate!($nav, $Δt, $f!, $p, $Q)
    return nothing
end


testKalmanAllocs()
testODEallocs()
testUDallocs()

testSigmaUKFallocs()
testUKFpropAllocs()
