using KalmanFilterEngine
using LinearAlgebra

function kalmanUpdateSimple(x, P, y, yEst, R, H, ns)
    K = (P * H') / (H * P * H' + R)
    K[ns+1:end, :] .= 0
    x = x + K * (y - yEst)
    P = (I - K * H) * P * (I - K * H)' + K * R * K'
    return x, P
end

# Reference: Algorithm 3.1 of Navigation Filter Best Practices
function kalmanUpdateSimpleScalar(x, P, y, yEst, R, H, ns; useP0=false)
    dx = zero(x)
    P0 = copy(P)
    for j in eachindex(y)
        Hj, Rj = H[j, :], R[j, j]
        if useP0
            Pxy = P0 * Hj
        else
            Pxy = P * Hj
        end
        Pyy = dot(Hj, Pxy) + Rj
        K = Pxy / Pyy
        K[ns+1:end] .= 0
        dx += K * (y[j] - yEst[j] - Hj' * dx)
        P = (I - K * Hj') * P * (I - K * Hj')' + K * Rj * K'
    end
    x .+= dx
    return x, P
end

function testUpdateGOAT()
    nx = 6
    ns = 6
    x0 = randn(nx)
    P0 = generatePosDefMatrix(nx)

    ny = 3
    H = randn(ny, nx)
    y = H * x0
    Rdiag = abs.(randn(ny))
    yMeas = y + Rdiag.*randn(ny)
    R = diagm(Rdiag)

    xTrue, PTrue = kalmanUpdateSimple(copy(x0), copy(P0), y, yMeas, R, H, ns)
    x1, P1 = kalmanUpdateSimpleScalar(copy(x0), copy(P0), y, yMeas, R, H, ns; useP0=true)
    x2, P2 = kalmanUpdateSimpleScalar(copy(x0), copy(P0), y, yMeas, R, H, ns; useP0=false)
    println("using P0 (Giacomo): $(norm(x1 - xTrue) + norm(P1 - PTrue))")
    println("using P (NASA): $(norm(x2 - xTrue) + norm(P2 - PTrue))")
    return nothing
end

testUpdateGOAT()
