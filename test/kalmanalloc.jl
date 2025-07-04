using KalmanFilterEngine
using LinearAlgebra
using BenchmarkTools

@views function kalmanUpdateErrorScalarNew!(nav, y, ŷ, R, H)
    δy = zero(y); δz = zero(y)
    isRejected = false

    for i in eachindex(y)
        # Estimated measurement and jacobians
        Pxy = nav.P*H[i, :]
        Pyy = H[i, :]'*Pxy + R[i, i]

        # Measurement editing
        δy[i] = y[i] - (ŷ[i] + H[i, :]'*nav.δx)
        δz[i] = δy[i]/sqrt(Pyy)                     # Normalized innovation
        isRejected = abs(δz[i]) > nav.σᵣ            # σ rejection threshold

        # Update error state and covariance matrix
        if !isRejected
            # Error state update
            Ks = Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
            nav.δx[1:nav.ns] .+= Ks*δy[i]

            # Covariance update (non-optimal gain with consider states)
            nav.P[1:nav.ns, 1:nav.ns] .-= Ks*Pyy*Ks'
            nav.P[1:nav.ns, nav.ns+1:nav.nδ] .-= Ks*Pxy[nav.ns+1:nav.nδ, :]'
            nav.P[nav.ns+1:nav.nδ, 1:nav.ns] = nav.P[1:nav.ns, nav.ns+1:nav.nδ]'
        end
    end

    return δy, δz, isRejected
end

nav = NavState(0.0, randn(8), generatePosDefMatrix(8))
nav.ns = 6

nav2 = deepcopy(nav)

y = [0.314; 12.00234; -3.3023]
yest = [0.214; 7.1234; -2.343]
R = 1e-3*I
H = [I zeros(3, 5)]

KalmanFilterEngine.kalmanUpdateErrorScalar!(nav, y, yest, R, H)
kalmanUpdateErrorScalarNew!(nav2, y, yest, R, H)
@show norm(nav.P - nav2.P) + norm(nav.x - nav2.x)

@btime KalmanFilterEngine.kalmanUpdateErrorScalar!($nav, $y, $yest, $R, $H)
@btime kalmanUpdateErrorScalarNew!($nav2, $y, $yest, $R, $H)
