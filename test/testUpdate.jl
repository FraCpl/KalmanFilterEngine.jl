using KalmanFilterEngine
using LinearAlgebra

function kalmanUpdateSimple(x, P, yMeas, yEst, R, H, ns)
    K = (P * H') / (H * P * H' + R)
    K[ns+1:end, :] .= 0
    x = x + K * (yMeas - yEst)
    P = (I - K * H) * P * (I - K * H)' + K * R * K'
    return x, P
end

function testUpdate(scalarUpdate=false)
    nx = 6; ns = 4
    x0 = randn(nx)
    P0 = generatePosDefMatrix(nx)
    nav = NavState(0.0, x0, P0, ns=ns)

    ny = 3
    H = randn(ny, nx)
    y = H * x0
    yMeas = copy(y) + randn(ny)

    if scalarUpdate
        R = diagm(abs.(randn(ny)))
        meas = NavMeasurementScalar(nx, ny; H=H, R=R, nReject=1000)
    else
        R = generatePosDefMatrix(ny)
        meas = NavMeasurement(nx, ny; H=H, R=R, nReject=1000)
    end

    meas.y .= y
    kalmanUpdate!(nav, yMeas, meas)
    xu, Pu = kalmanUpdateSimple(copy(x0), copy(P0), yMeas, y, R, H, ns)

    @show errx = norm(xu - nav.x)
    @show errp = norm(Pu - nav.P)
    return errx + errp < 1e-14
end

# @show testUpdate(true)
# @show testUpdate(false)
