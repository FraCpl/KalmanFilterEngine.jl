using KalmanFilterEngine
using LinearAlgebra

# Naive implementation of Schmidt-Kalman filter batch update
function kalmanUpdateSimple(x, P, yMeas, yEst, R, H, ns)
    K = (P * H') / (H * P * H' + R)
    K[ns+1:end, :] .= 0
    x = x + K * (yMeas - yEst)
    P = (I - K * H) * P * (I - K * H)' + K * R * K'
    return x, P
end

function testUpdate(mode=1)
    nx = 6; ns = 4
    x0 = randn(nx)
    P0 = generatePosDefMatrix(nx)
    nav = NavState(0.0, x0, P0, ns=ns)

    ny1 = 3
    H1 = randn(ny1, nx)
    y1 = H1 * x0
    yMeas1 = y1 + randn(ny1)

    ny2 = 4
    H2 = randn(ny2, nx)
    y2 = H2 * x0
    yMeas2 = y2 + randn(ny2)

    if mode == 1
        # Scalar measurements
        R1 = diagm(abs.(randn(ny1)))
        meas1 = NavMeasurementScalar(nx, ny1; H=H1, R=R1, nReject=1000)
        R2 = diagm(abs.(randn(ny2)))
        meas2 = NavMeasurementScalar(nx, ny2; H=H2, R=R2, nReject=1000)
    elseif mode == 2
        # Fully-correlated measurements
        R1 = generatePosDefMatrix(ny1)
        meas1 = NavMeasurement(nx, ny1; H=H1, R=R1, nReject=1000)
        R2 = generatePosDefMatrix(ny2)
        meas2 = NavMeasurement(nx, ny2; H=H2, R=R2, nReject=1000)
    else
        # One scalar and one correlated measurement
        R1 = diagm(abs.(randn(ny1)))
        meas1 = NavMeasurementScalar(nx, ny1; H=H1, R=R1, nReject=1000)
        R2 = generatePosDefMatrix(ny2)
        meas2 = NavMeasurement(nx, ny2; H=H2, R=R2, nReject=1000)
    end

    meas1.y .= y1
    meas2.y .= y2

    # Sequential Kalman update
    nav.δx .= 0.0
    kalmanUpdateError!(nav, yMeas1, meas1)
    kalmanUpdateError!(nav, yMeas2, meas2)
    kalmanErrorToFullState!(nav)

    # Batch update of two measurements with naive Kalman filter formulas
    R = [R1 zeros(ny1, ny2); zeros(ny2, ny1) R2]
    xu, Pu = kalmanUpdateSimple(copy(x0), copy(P0), [yMeas1; yMeas2], [y1; y2], R, [H1; H2], ns)

    @show errx = norm(xu - nav.x)
    @show errp = norm(Pu - nav.P)
    return errx + errp < 1e-13
end

@show "Scalar meas", testUpdate(1)
@show "Full meas", testUpdate(2)
@show "Mix meas", testUpdate(3)
