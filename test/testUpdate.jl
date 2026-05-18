using KalmanFilterEngine
using LinearAlgebra

function kalmanUpdateSimple(x, P, y, yEst, R, H, ns)
    Pxy = P * H'
    Pyy = R + H * Pxy
    K = Pxy / Pyy
    K[ns+1:end, :] .= 0
    x = x + K * (y - yEst)
    P = (I - K * H) * P * (I - K * H)' + K * R * K'
    return x, P
end

# Reference: Algorithm 3.1 of Navigation Filter Best Practices
function kalmanUpdateSimpleScalar(x, P, y, yEst, R, H, ns)
    dx = zero(x)
    for i in eachindex(y)
        Pxy = P * H[i, :]
        Pyy = R[i, i] + dot(H[i, :], Pxy)
        K = Pxy / Pyy
        K[ns+1:end] .= 0
        dx += K * (y[i] - yEst[i] - H[i, :]' * dx)
        P = (I - K * H[i, :]') * P * (I - K * H[i, :]')' + K * R[i, i] * K'
    end
    x .+= dx
    return x, P
end

function testUpdate(scalarUpdate=false)
    nx = 6; ns = 3
    x0 = randn(nx)
    P0 = generatePosDefMatrix(nx)
    nav = NavState(0.0, x0, P0, ns=ns)

    ny = 3
    y = randn(ny)

    if scalarUpdate
        meas = NavMeasurementScalar(nx, ny; H=randn(ny, nx), R=diagm(abs.(randn(ny))), nReject=1000)
        meas.y .= randn(ny)
        kalmanUpdate!(nav, y, meas)
        xu, Pu = kalmanUpdateSimpleScalar(x0, P0, y, meas.y, meas.R, meas.H, nav.ns)
    else
        meas = NavMeasurement(nx, ny; H=randn(ny, nx), R=generatePosDefMatrix(ny), nReject=1000)
        meas.y .= randn(ny)
        kalmanUpdate!(nav, y, meas)
        xu, Pu = kalmanUpdateSimple(x0, P0, y, meas.y, meas.R, meas.H, nav.ns)
    end

    @show err = norm(xu - nav.x) + norm(Pu - nav.P)
    return err < 1e-14
end
