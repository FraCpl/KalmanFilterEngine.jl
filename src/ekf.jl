mutable struct NavStateEKF{T<:AbstractVector{Float64},M<:AbstractMatrix{Float64},D<:AbstractVector{Float64}} <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x::T                    # Full estimated state, x[t]
    P::M                    # Covariance Matrix, P[t]
    δx::D                   # Error state, δx[t]
    const ns::Int64         # Number of solve for (error) states
    const nc::Int64         # Number of consider (error) states
    const nδ::Int64         # Number of error states
    tc::Float64             # Nav time of Pcc
    Pcc::Matrix{Float64}    # Consider-covariance matrix Pcc, where P = [Pss Psc; Pcs Pcc]
    odeCache::ODECache{T, M}
end

function NavStateEKF(t, x, P, ns=size(P, 1))
    nδ = size(P, 1)     # number of error states
    odeCache = ODECache(x, P)
    return NavStateEKF(t, x, P, zero(P[:, 1]), ns, nδ - ns, nδ, t - 1.0, zero(P[ns+1:nδ, ns+1:nδ]), odeCache)
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
    kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps=1)

Propagate navigation state forward in time for ```Δt``` time units.

Inputs include the dynamics function ```f!(ẋ, x, p, t)```, dynamics jacobian
function ```Jf!(Fx, x, p, t)```, and equivalent discrete-time process noise
covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
number of RK steps to be performed when numerically integrating the system's
dynamics.
"""
function kalmanPropagate!(nav::NavStateEKF, Δt, f!, Jf!, p, Q; nSteps=1)
    odeSolve!(nav.x, nav.t, Δt, f!, Jf!, p, nav.odeCache; nSteps=nSteps)
    kalmanPropagateCov!(nav, nav.odeCache.Φ, Q)
    nav.t += Δt
    return nothing
end


"""
    kalmanPropagateCov!(nav::NavStateEKF, Φ, Q)

Propagate the state covariance matrix of an Extended Kalman Filter (EKF).

This function implements the standard discrete-time covariance propagation:

    Pₖ₊₁ = Φ * Pₖ * Φᵀ + Q

where:
- `Φ` is the state transition matrix between tₖ and tₖ₊₁,
- `Pₖ` is the current covariance,
- `Q` is the (discrete time) process noise covariance.

# Arguments
- `nav::NavStateEKF`: EKF navigation state (modified in-place).
- `Φ`: State transition matrix.
- `Q`: Process noise covariance matrix.

# Notes
- This function assumes `Φ` and `Q` are dimensionally consistent with `nav.P`.
- Symmetry and positive-definiteness of `P` are not enforced explicitly;
  ensure `Q` is positive semi-definite and `Φ` is well-conditioned.
"""
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
    kalmanUpdate!(nav, y, meas)

Update state of the Kalman filter using the input measurement.
"""
function kalmanUpdate!(nav::NavStateEKF, y, meas::M) where {M<:AbstractNavMeasurement}
    nav.δx .= 0.0       # Better safe than sorry
    isRejected = kalmanUpdateError!(nav, y, meas)
    kalmanErrorToFullState!(nav)
    return isRejected
end

"""
    kalmanUpdate!(nav, y, h!, meas, p, t)

h!(meas, x, p, t)
Update state of the Kalman filter using the input measurement.
"""
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
    nc = nav.nc
    ny = length(y)
    δx = nav.δx
    P = nav.P

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

    # Update error state and covariance matrix
    # We update the full error state and covariance matrix with the optimal filter gain and
    # covariance update formulas. Consider states are handled, only after all synchronized
    # measurements have been processed, by calling kalmanErrorToFullState!
    @inbounds for r in 1:nδ, j in 1:ny
        # Kalman gain
        k = K[r, j]

        # Error state
        δx[r] += k * δy[j]

        # Covariance update: P -= K * Pxy'
        for c in r:nδ
            P[r, c] -= k * Pxy[c, j]
            P[c, r] = P[r, c]
        end
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
    nc = nav.nc
    δx = nav.δx
    P = nav.P

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
        # We update the full error state and covariance matrix with the optimal filter gain and
        # covariance update formulas. Consider states are handled, only after all synchronized
        # measurements have been processed, by calling kalmanErrorToFullState!
        iPyy = 1 / Pyy
        @inbounds for r in 1:nδ
            # Kalman gain (scalar measurement)
            K = Pxy[r] * iPyy

            # State update
            δx[r] += K * δy[i]

            # Covariance update: P -= K * Pxy'
            for c in r:nδ
                P[r, c] -= K * Pxy[c]
                P[c, r] = P[r, c]   # maintain symmetry explicitly
            end
        end
    end

    return false
end

# For some reason the default function sum! provides wrong result
@inline function sumDefault!(x, δx, p)
    @inbounds for i in eachindex(x)
        x[i] += δx[i]
    end
end

# sumState!(x, δx, p), updates x with x ⨁ δx
function kalmanErrorToFullState!(nav::NavStateEKF, sumState!::F=sumDefault!, p::T=nothing) where {F, T}
    # The error state has been computed using the optimal Kalman formulas on the full error
    # state, i.e., without separately considering solve-for and consider error states.
    # In this function we restore the correct Schmidt-Kalman formulation, by setting to
    # zero the consider error states (so that the consider full states are NOT updated), and
    # restoring the original consider-consider covariance matrix Pcc.
    # When manually implementing an ESKF, this function MUST be called at the end of the
    # measurements updates cycle.
    ns = nav.ns
    nδ = nav.nδ
    nc = nav.nc

    # Set to zero consider parameters
    @inbounds for i in ns+1:nδ
        nav.δx[i] = 0
    end
    sumState!(nav.x, nav.δx, p)

    # Restore original covariance matrix for consider-parameters
    @inbounds for i in 1:nc, j in i:nc
        nav.P[i+ns, j+ns] = nav.P[j+ns, i+ns] = nav.Pcc[i, j]
    end

    # Reset time flag and error state
    nav.tc = nav.t - 1      # We only need tc to be smaller than t
    nav.δx .= 0             # better safe than sorry

    return nothing
end
