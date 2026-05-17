# This update routine implements an IEKF
# https://ntrs.nasa.gov/api/citations/20140006041/downloads/20140006041.pdf
function kalmanUpdateIter!(nav::NavStateEKF, y, h!, meas::NavMeasurement=NavMeasurement(nav.nδ, length(y)), p=nothing, t=nothing; iter=3)

    ns = nav.ns
    nδ = nav.nδ
    xIter = nav.odeCache.K1
    δx = nav.δx
    P = nav.P
    x = nav.x
    ny = length(y)
    δy = meas.δy
    δz = meas.δz
    Pxy = meas.Pxy
    Pyy = meas.Pyy
    K = meas.K
    nReject = meas.nReject

    # Start iterations
    xIter .= x
    @inbounds for itr in 1:iter
        # Estimated measurement and jacobians
        h!(meas, xIter, p, t)
        ŷ = meas.y
        R = meas.R
        H = meas.H
        mul!(Pxy, P, transpose(H))
        mul!(Pyy, H, Pxy)
        Pyy .+= R

        # Measurement editing
        if itr == 1
            for i in eachindex(y)
                # Check negative covariance (numerical issue)
                Pyy[i, i] ≤ 0 && return true

                # Innovation and normalized innovation
                δy[i] = y[i] - ŷ[i]
                δz[i] = δy[i] / sqrt(Pyy[i, i])

                # Check rejection threshold
                abs(δz[i]) > nReject && return true
            end
        end

        # Compute Kalman gain
        K .= Pxy
        rdiv!(K, cholesky!(Hermitian(Pyy)))        # K = Pxy / Pyy, Caution: this modifies Pyy

        # State update
        # xIter[1:nav.ns] .= nav.x[1:nav.ns] + K[1:ns, :]*(y - ŷ - H*(nav.x - xIter))
        δx .= x .- xIter
        xIter .= x
        for j in 1:ny
            # compute Hδx once
            Hδxj = 0.0
            for k in 1:nδ
                Hδxj += H[j, k] * δx[k]
            end

            # rank-1 update
            rj = y[j] - ŷ[j] - Hδxj
            for r in 1:ns
                xIter[r] += K[r, j] * rj
            end
        end
    end

    # Update state
    x .= xIter
    xIter .= 0      # Reset ODE cache variables

    # Update covariance matrix (non-optimal gain with consider states)
    @inbounds for r in 1:ns, j in 1:ny
        # P[1:ns, 1:ns] .-= Ks * Pyy * Ks' = -Pxy * Ks'
        # P[1:ns, (ns + 1):nδ] .-= Ks * Pyx
        pxy = Pxy[r, j]
        k = K[r, j]

        # Top left block: P[1:ns, 1:ns] (upper triangular only)
        for c in r:ns
            P[r, c] -= pxy * K[c, j]
        end

        # Top right block: P[1:ns, (ns + 1):nδ]
        for c in ns+1:nδ
            P[r, c] -= k * Pxy[c, j]
        end
    end

    # Make covariance matrix symmetric
    # P[1:ns, 1:ns] (lower triangular only)
    @inbounds for r in 2:ns, c in 1:r-1
        P[r, c] = P[c, r]
    end
    # P[ns+1:nδ, 1:ns]
    @inbounds for r in (ns + 1):nδ, c in 1:ns
        P[r, c] = P[c, r]
    end

    return false
end
