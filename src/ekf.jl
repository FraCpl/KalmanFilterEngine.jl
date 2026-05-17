mutable struct NavStateEKF{T<:AbstractVector{Float64},M<:AbstractMatrix{Float64},D<:AbstractVector{Float64}} <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x::T                    # Full estimated state, x[t]
    P::M                    # Covariance Matrix, P[t]
    δx::D                   # Error state, δx[t]
    const ns::Int64         # Number of solve for (error) states
    const nδ::Int64         # Number of error states
    odeCache::ODECache{T, M}
end

"""
    NavStateEKF(t, x, P)

Build EKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateEKF(t, x, P, ns=size(P, 1))
    nδ = size(P, 1)
    odeCache = ODECache(x, P)
    return NavStateEKF(t, x, P, zero(P[:, 1]), ns, nδ, odeCache)
end

"""
    getCov(nav)

Get navigation covariance matrix ``P``.
"""
getCov(nav::NavStateEKF) = nav.P

# This is the Kalman filter propagation routine for a continuous time
# dynamical model described by a set of 1st order ordinary differential
# equations.
"""
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = 1)

Propagate navigation state forward in time for ```Δt``` time units.

Inputs include the dynamics function ```f!(ẋ, x, p, t)```, dynamics jacobian
function ```Jf!(Fx, x, p, t)```, and equivalent discrete-time process noise
covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
number of RK4 steps to be performed when numerically integrating the system's
dynamics. This function is only applicable to EKF and UDEKF.
"""
function kalmanPropagate!(nav::NavStateEKF, Δt, f!, Jf!, p, Q; nSteps=1)
    odeSolve!(nav.x, nav.t, Δt, f!, Jf!, p, nav.odeCache; nSteps=nSteps)
    kalmanPropagateCov!(nav, nav.odeCache.Φ, Q)
    nav.t += Δt
    return nothing
end

# This function implements the covariance propagation formula
# P[k+1] = ϕ*P[k]*ϕᵀ + Q
function kalmanPropagateCov!(nav::NavStateEKF, Φ, Q)
    transformCov!(nav, Φ)
    nav.P .+= Q
    return nav.P
end

# P' = A*P*Aᵀ, where size(A) == size(P)
@inline function transformCov!(nav::NavStateEKF, A)
    Aᵀ = nav.odeCache.P2
    PAᵀ = nav.odeCache.P1
    transpose!(Aᵀ, A)
    mul!(PAᵀ, nav.P, Aᵀ)
    mul!(nav.P, A, PAᵀ)
    return nav.P
end

"""
    kalmanUpdate!(nav, meas, y)

Update state of the Kalman filter using the input measurement.
"""
function kalmanUpdate!(nav::NavStateEKF, y, meas::M) where {M<:AbstractNavMeasurement}
    nav.δx .= 0.0       # Better safe than sorry
    isRejected = kalmanUpdateError!(nav, y, meas)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state
    return isRejected
end

function kalmanUpdate!(nav::NavStateEKF, y, h!, meas::M=NavMeasurement(nav.nδ, length(y)), p=nothing, t=nothing) where {M<:AbstractNavMeasurement}
    h!(meas, nav.x, p, t)
    return kalmanUpdate!(nav, y, meas)
end

function kalmanUpdateError!(nav::NavStateEKF, y, h!, meas::M=NavMeasurement(nav.nδ, length(y)), p=nothing, t=nothing) where {M<:AbstractNavMeasurement}
    h!(meas, nav.x, p, t)
    return kalmanUpdateError!(nav, y, meas)
end

"""
    kalmanUpdateError!(nav, y, meas)

Update error state of the Kalman filter using the input measurement.
This function is only applicable to EKF and UDEKF.
"""
# Returns an 'isRejected' flag.
# Recursive Implementations of the Schmidt-Kalman Consider Filter (Zanetti, D'Souza)
function kalmanUpdateError!(nav::NavStateEKF, y, meas::NavMeasurement)

    # Extract data from nav and meas
    ŷ = meas.y
    R = meas.R
    H = meas.H
    δy = meas.δy
    δz = meas.δz
    Pxy = meas.Pxy
    Pyy = meas.Pyy
    K = meas.K
    nReject = meas.nReject

    ns = nav.ns
    nδ = nav.nδ
    ny = length(y)
    δx = nav.δx
    P = nav.P

    # Estimated measurement and jacobians
    mul!(Pxy, P, transpose(H))      # Pxy = P*Hᵀ
    mul!(Pyy, H, Pxy)               # Pyy = H*P*Hᵀ + R
    Pyy .+= R

    # Measurement editing
    # δy := y - (ŷ + H*δx)
    mul!(δy, H, δx)     # This really is H*δx here
    @inbounds for i in eachindex(y)
        # Check negative covariance (numerical issue)
        Pyy[i, i] ≤ 0 && return true

        # Innovation and normalized innovation
        δy[i] = y[i] - ŷ[i] - δy[i]     # Fix innovation definition wrt mul!()
        δz[i] = δy[i] / sqrt(Pyy[i, i])

        # Check rejection threshold
        abs(δz[i]) > nReject && return true
    end

    # Compute Kalman Gain
    K .= Pxy
    rdiv!(K, cholesky!(Hermitian(Pyy)))        # K = Pxy / Pyy, Caution: this modifies Pyy

    # Update error state and covariance matrix (non-optimal gain with consider states)
    @inbounds for r in 1:ns, j in 1:ny
        # P[1:ns, 1:ns] .-= Ks * Pyy * Ks' = -Pxy * Ks'
        # P[1:ns, (ns + 1):nδ] .-= Ks * Pyx
        pxy = Pxy[r, j]
        k = K[r, j]

        # Error state
        δx[r] += k * δy[j]

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

# The following function can be directly used when R is a diagonal matrix
# Returns an 'isRejected' flag.
function kalmanUpdateError!(nav::NavStateEKF, y, meas::NavMeasurementScalar)

    # Extract data from nav and meas
    ŷ = meas.y
    R = meas.R
    H = meas.H
    δy = meas.δy
    δz = meas.δz
    Pxy = meas.Pxy
    nReject = meas.nReject
    nδ = nav.nδ
    ns = nav.ns
    δx = nav.δx
    P = nav.P

    # Cycle through each scalar component of the measurement vector
    @inbounds for i in eachindex(y)
        # Compute Pxy, Pyy, and Hδx
        # Pxy = P * H[i, :]'
        # Pyy = H[i, :] * P * H[i, :]' + R
        Pyy = R[i, i]
        Hδx = 0.0

        for j in 1:nδ
            hij = H[i, j]
            Hδx += hij * δx[j]

            acc = 0.0
            for k in 1:nδ
                acc += P[j, k] * H[i, k]
            end

            Pxy[j] = acc
            Pyy += hij * acc
        end

        # Check measurement rejection because of numerical errors
        Pyy ≤ 0 && return true

        # Measurement editing
        δy[i] = y[i] - (ŷ[i] + Hδx)
        δz[i] = δy[i] / sqrt(Pyy)                   # Normalized innovation
        abs(δz[i]) > nReject && return true         # σ rejection threshold

        # Update error state and covariance matrix
        iPyy = 1 / Pyy
        @inbounds for r in 1:ns
            # Kalman Gain
            K = Pxy[r] * iPyy

            # Error state update, δx = K * δy
            δx[r] += K * δy[i]

            # Update covariance matrix (non-optimal gain with consider states)
            # P[1:ns, 1:ns] -= Ks * Pyy * Ks'
            # P[1:ns, ns+1:nδ] -= Ks * Pxy[ns+1:nδ, :]'
            # For 1:ns 1:ns terms: # K * Pyy * Kc = K * Pyy * Pxy[c] / Pyy = K * Pxy[c]
            for c in r:nδ
                P[r, c] -= K * Pxy[c]
                P[c, r] = P[r, c]           # Make covariance matrix symmetric
            end
        end
    end

    return false
end
