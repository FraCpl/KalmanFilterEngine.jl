# This update routine implements an IEKF
# https://ntrs.nasa.gov/api/citations/20140006041/downloads/20140006041.pdf
function kalmanUpdateIter!(nav::NavStateEKF, y, h!, meas::NavMeasurement=NavMeasurement(nav.nx, length(y)), p=nothing, t=nothing; iter=3)
    # @warn "TO BE UPDATED! --> SEE ESKF"
    ns = nav.ns
    nx = nav.nx
    nc = nav.nc
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

    # If Pcc has not been saved yet, save it.
    # This is required when updating the full state with the error state once all
    # simultaneous measurements have been processed
    if nc > 0 && nav.tc < nav.t
        # We don't store the exact time to avoid float comparison, we just make sure that
        # tc is now bigger than t.
        nav.tc = nav.t + 1
        @inbounds for i in 1:nc, j in i:nc
            # We only save upper triangular as it is the only one used in kalmanErrorToFullState!
            nav.Pcc[i, j] = P[i+ns, j+ns]
        end
    end

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
            for k in 1:nx
                Hδxj += H[j, k] * δx[k]
            end

            # rank-1 update
            rj = y[j] - ŷ[j] - Hδxj
            for r in 1:nx
                xIter[r] += K[r, j] * rj
            end
        end
    end

    # Update state
    @inbounds for i in 1:ns
        x[i] = xIter[i]     # Only solve-for states are updated
    end
    xIter .= 0      # Reset ODE cache variables

    # Update covariance matrix
    # We update the full covariance matrix with the optimal filter gain and
    # covariance update formulas. Consider states are handled, only after all synchronized
    # measurements have been processed, by calling kalmanErrorToFullState!
    @inbounds for r in 1:nx, j in 1:ny
        # Kalman gain
        k = K[r, j]

        # Covariance update: P -= K * Pxy'
        for c in r:nx
            P[r, c] -= k * Pxy[c, j]
            P[c, r] = P[r, c]
        end
    end

    # Restore original covariance matrix for consider-parameters
    @inbounds for i in 1:nc, j in i:nc
        nav.P[i+ns, j+ns] = nav.P[j+ns, i+ns] = nav.Pcc[i, j]
    end
    nav.tc = nav.t - 1      # We only need tc to be smaller than t

    return false
end
