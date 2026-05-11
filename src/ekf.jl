mutable struct NavStateEKF{T<:AbstractVector{Float64},M<:AbstractMatrix{Float64},D<:AbstractVector{Float64}} <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x::T                    # Full estimated state, x[t]
    P::M                    # Covariance Matrix, P[t]
    δx::D                   # Error state, δx[t]
    const ns::Int64         # Number of solve for (error) states
    const nδ::Int64         # Number of error states

    # Internal allocation variables
    KPyx::Matrix{Float64}
    xs::Vector{Float64}
    pxy::Vector{Float64}
    odeCache::ODECache
end

"""
    NavStateEKF(t, x, P)

Build EKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateEKF(t, x, P, ns=size(P, 1))
    nδ = size(P, 1)
    odeCache = ODECache(x, P)
    return NavStateEKF(t, x, P, zero(P[:, 1]), ns, nδ, zeros(ns, nδ - ns), zeros(ns), zeros(nδ), odeCache)
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
    _, Φ = odeSolve!(nav.x, nav.t, Δt, f!, Jf!, p, nav.odeCache; nSteps=nSteps)
    kalmanPropagateCov!(nav, Φ, Q)
    nav.t += Δt
    return nothing
end

# This function implements the covariance propagation formula
# P[k+1] = ϕ*P[k]*ϕᵀ + Q
function kalmanPropagateCov!(nav::NavStateEKF, Φ, Q)
    tmp = nav.odeCache.P1
    mul!(tmp, nav.P, transpose(Φ))
    mul!(nav.P, Φ, tmp)
    nav.P .+= Q
    return nav.P
end

# """
#     kalmanPropagate!(nav, Δt, f, Q; nSteps = 1)

# Propagate navigation state forward in time for ```Δt``` time units.

# Inputs include the dynamics function ```ẋ = f(t, x)```, and equivalent discrete-time process noise
# covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
# number of RK4 steps to be performed when numerically integrating the system's
# dynamics.
# """
# function kalmanPropagate!(nav::NavStateEKF, Δt, f, Q; nSteps=1)
#     Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
#     kalmanPropagate!(nav, Δt, f, Jf, Q, nSteps = nSteps)
# end

"""
    kalmanUpdate!(nav, t, y, h)

Update state of the Kalman filter using the input measurement.

Inputs include the measurement time ```t```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x)```. When using SRUKF or UKF, the
measurement function only needs to provide ```ŷ``` and ```R``` as outputs.
"""
function kalmanUpdate!(nav::NavStateEKF, t, y, h; nReject::Int=6)
    δy, δz, isRejected = kalmanUpdateError!(nav, t, y, h; nReject=nReject)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state

    return δy, δz, isRejected
end

function kalmanUpdate!(nav::NavStateEKF, y, ŷ, R, H,
    δy=zero(y), δz=zero(y), Pxy=Matrix{eltype(nav.P)}(undef, nav.nδ, length(y)), Pyy=Matrix{eltype(nav.P)}(undef, size(R));     # Save allocations
    nReject::Int=6)

    nav.δx .= 0.0       # Better safe than sorry
    isRejected = kalmanUpdateError!(nav, y, ŷ, R, H, δy, δz, Pxy, Pyy; nReject=nReject)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state

    return isRejected
end

@inline function kalmanUpdateScalar!(nav::NavStateEKF, t, y, h; nReject::Int=6)
    nav.δx .= 0.0       # Better safe than sorry
    δy, δz, isRejected = kalmanUpdateErrorScalar!(nav, t, y, h; nReject=nReject)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state

    return δy, δz, isRejected
end

@inline function kalmanUpdateScalar!(nav::NavStateEKF, y, ŷ, R, H,
    δy=zero(y), δz=zero(y); nReject::Int=6)

    nav.δx .= 0.0       # Better safe than sorry
    isRejected = kalmanUpdateErrorScalar!(nav, y, ŷ, R, H, δy, δz; nReject=nReject)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state

    return isRejected
end

"""
    kalmanUpdateError!(nav, t, y, h)

Update error state of the Kalman filter using the input measurement.

Inputs include the measurement time ```t```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x)```. This function is only applicable
to EKF and UDEKF.
"""
@inline function kalmanUpdateError!(nav::NavStateEKF, t, y, h; nReject::Int=6)
    # Predict measurement, and compute noise covariance matrix and jacobian
    ŷ, R, H = h(t, nav.x)

    # Allocate innovation and normalized innovation
    δy = zero(y)
    δz = zero(y)

    # Perform kalman update
    isRejected = kalmanUpdateError!(nav, y, ŷ, R, H, δy, δz; nReject=nReject)

    # Return results
    return δy, δz, isRejected
end

# Scalar measurement update for EKF
# The following function can be directly used when R is a diagonal matrix
@inline function kalmanUpdateErrorScalar!(nav::NavStateEKF, t, y, h; nReject::Int=6)
    ŷ, R, H = h(t, nav.x)
    δy = zero(y)
    δz = zero(y)

    isRejected = kalmanUpdateErrorScalar!(nav, y, ŷ, R, H, δy, δz; nReject=nReject)

    return δy, δz, isRejected
end

# Returns an 'isRejected' flag.
# Recursive Implementations of the Schmidt-Kalman Consider Filter (Zanetti, D'Souza)
function kalmanUpdateError!(nav::NavStateEKF, y, ŷ, R, H,
    δy=zero(y),                                                       # Save allocations
    δz=zero(y),                                                       # Save allocations
    Pxy=Matrix{eltype(nav.P)}(undef, nav.nδ, length(y)),              # Save allocations
    Pyy=Matrix{eltype(nav.P)}(undef, size(R));                        # Save allocations
    nReject::Int=6
)

    ns = nav.ns
    nδ = nav.nδ
    ny = length(y)

    # Estimated measurement and jacobians
    mul!(Pxy, nav.P, transpose(H))  # Pxy = P*Hᵀ
    mul!(Pyy, H, Pxy)               # Pyy = H*P*Hᵀ + R
    Pyy .+= R

    # Measurement editing
    # δy := y - (ŷ + H*δx)
    mul!(δy, H, nav.δx)     # This really is H*δx here
    @inbounds for i in eachindex(y)
        # Check negative covariance (numerical issue)
        Pyy[i, i] ≤ 0 && return true

        # Innovation and normalized innovation
        δy[i] = y[i] - ŷ[i] - δy[i]     # Fix innovation definition wrt mul!()
        δz[i] = δy[i] / sqrt(Pyy[i, i])

        # Check rejection threshold
        abs(δz[i]) > nReject && return true
    end

    # Update error state and covariance matrix
    # Compute Kalman Gain
    Ks = Pxy[1:nav.ns, :] / Pyy
    mul!(nav.xs, Ks, δy)            # Error state correction

    @inbounds for i in 1:ns
        # Update error state
        nav.δx[i] += nav.xs[i]

        # Covariance update (non-optimal gain with consider states)
        # P[1:ns, 1:ns] .-= Ks * Pyy * Ks'
        for c in 1:ns
            acc = 0.0
            for k in 1:ny, l in 1:ny
                acc += Ks[i, k] * Pyy[k, l] * Ks[c, l]
            end
            nav.P[i, c] -= acc
        end

        # P[1:ns, (ns + 1):nδ] .-= Ks * Pyx
        for c in (ns+1):nδ
            acc = 0.0
            for k in 1:ny
                acc += Ks[i, k] * Pxy[c, k]
            end
            nav.P[i, c] -= acc
        end
    end

    # Make covariance matrix symmetric
    # P[ns+1:nδ, 1:ns] = P[1:ns, ns+1:nδ]'
    @inbounds for r in (ns + 1):nδ, c in 1:ns
        nav.P[r, c] = nav.P[c, r]       # Make it symmmetric
    end

    return false
end

# The following function can be directly used when R is a diagonal matrix
# Returns an 'isRejected' flag.
function kalmanUpdateErrorScalar!(nav::NavStateEKF, y, ŷ, R, H, δy=zero(y), δz=zero(y); nReject::Int=6)

    # Extract data from nav
    Pxy = nav.pxy
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
            @simd for k in 1:nδ
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
        @inbounds for j in 1:ns
            # Kalman Gain
            Kj = Pxy[j] * iPyy

            # Error state update, δx = K * δy
            δx[j] += Kj * δy[i]

            # Update covariance matrix (non-optimal gain with consider states)
            # P[1:ns, 1:ns] -= Ks * Pyy * Ks'
            # P[1:ns, ns+1:nδ] -= Ks * Pxy[ns+1:nδ, :]'
            # For 1:ns 1:ns terms: # Kj * Pyy * Kc = Kj * Pyy * Pxy[c] / Pyy = Kj * Pxy[c]
            for c in 1:nδ
                P[j, c] -= Kj * Pxy[c]
            end
        end

        # Make covariance matrix symmetric
        # P[ns+1:nδ, 1:ns] = P[1:ns, ns+1:nδ]'
        @inbounds for r in (ns + 1):nδ, c in 1:ns
            P[r, c] = P[c, r]
        end
    end

    return false
end

# This update routine implements an IEKF
# TODO: Fix, not super efficient
@views function kalmanUpdateIter!(nav::NavStateEKF, t, y, h, iter::Int=3,
    δy=zero(y), δz=zero(y), Pxy=Matrix{eltype(nav.P)}(undef, nav.nδ, length(y)), Pyy=Matrix{eltype(nav.P)}(undef, length(y), length(y)); nReject::Int=6)

    # ns = nav.ns
    # nδ = nav.nδ
    ny = length(y)
    xIter = nav.xs

    # Call function for first time
    ŷ, R, H = h(t, nav.x)

    # Estimated measurement and jacobians
    mul!(Pxy, nav.P, transpose(H))  # Pxy = P*Hᵀ
    mul!(Pyy, H, Pxy)               # Pyy = H*P*Hᵀ + R
    Pyy .+= R

    # Measurement editing
    # δy := y - (ŷ + H*δx)
    mul!(δy, H, nav.δx)     # This really is H*δx here
    @inbounds for i in eachindex(y)
        # Check negative covariance (numerical issue)
        Pyy[i, i] ≤ 0 && return δy, δz, true

        # Innovation and normalized innovation
        δy[i] = y[i] - ŷ[i] - δy[i]     # Fix innovation definition wrt mul!()
        δz[i] = δy[i] / sqrt(Pyy[i, i])

        # Check rejection threshold
        abs(δz[i]) > nReject && return δy, δz, true
    end

    # Update error state and covariance matrix
    Ks = zeros(nav.ns, ny)
    xIter .= nav.x

    # Start iterations
    @inbounds for i in 1:iter
        if i > 1
            ŷ, R, H = h(t, xIter)
            mul!(Pxy, nav.P, transpose(H))
            mul!(Pyy, H, Pxy)
            Pyy .+= R
        end

        # State update
        Ks .= Pxy[1:nav.ns, :] / Pyy    # Kalman Gain
        xIter[1:nav.ns] .= nav.x[1:nav.ns] + Ks*(y - ŷ - H*(nav.x - xIter))
    end

    # Update state
    nav.x .= xIter

    # Covariance update (non-optimal gain with consider states)
    nav.P[1:nav.ns, 1:nav.ns] .-= Ks*Pyy*Ks'
    mul!(nav.KPyx, Ks, transpose(Pxy[(nav.ns + 1):nav.nδ, :]))    # KPyx = Ks*Pxyᵀ
    nav.P[1:nav.ns, (nav.ns + 1):nav.nδ] .-= nav.KPyx             # In-place subtraction
    @inbounds for ir in (nav.ns + 1):nav.nδ, ic in 1:nav.ns
        nav.P[ir, ic] = nav.P[ic, ir]       # Make it symmmetric
    end

    return δy, δz, false
end
