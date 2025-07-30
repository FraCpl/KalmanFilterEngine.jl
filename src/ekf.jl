mutable struct NavStateEKF{T<:AbstractVector{Float64}, M<:AbstractMatrix{Float64}, D<:AbstractVector{Float64}} <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x::T                    # Full estimated state, x[t]
    P::M                    # Covariance Matrix, P[t]
    δx::D                   # Error state, δx[t]
    ns::Int64               # Number of solve for (error) states
    σᵣ::Int64               # Outlier rejection threshold
    nδ::Int64               # Number of error states
    iter::Int64             # Number of iterations for IEKF

    # Internal allocation variables
    KPyyK::Matrix{Float64}
    KPyx::Matrix{Float64}
    xs::Vector{Float64}
    pxy::Vector{Float64}
end

"""
    NavStateEKF(t, x, P)

Build EKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateEKF(t, x, P, ns=size(P, 1); iter=0)
    nδ = size(P, 1)
    return NavStateEKF(t, x, P, zero(P[:, 1]), ns, 6, nδ, iter,
        zeros(ns, ns), zeros(ns, nδ - ns), zeros(ns), zeros(nδ))
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

Inputs include the dynamics function ```ẋ = f(t, x)```, dynamics jacobian
function ```Fx = Jf(t, x)```, and equivalent discrete-time process noise
covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
number of RK4 steps to be performed when numerically integrating the system's
dynamics. This function is only applicable to EKF and UDEKF.
"""
function kalmanPropagate!(nav::NavStateEKF, Δt, f, Jf, Q; nSteps=1)
    Φ = kalmanPropagateState!(nav, Δt, f, Jf; nSteps=nSteps)
    kalmanPropagateCov!(nav, Φ, Q)
end

# This function propagates the full navigation state from the current
# time to the current time plus Δt using a Runge-Kutta algorithm. It
# also computes the state transition matrix by numerical integration
# of the Jacobian of the dynamics.
function kalmanPropagateState!(nav, Δt, f, Jf; nSteps=1)
    nav.x, Φ = odeCore(nav.t, nav.x, Matrix(1.0I, nav.nδ, nav.nδ), Δt, f, Jf; nSteps=nSteps)
    nav.t += Δt
    return Φ
end

# This function implements the covariance propagation formula
# P[k+1] = ϕ*P[k]*ϕᵀ + Q
function kalmanPropagateCov!(nav::NavStateEKF, Φ, Q, tmp=similar(nav.P))
    mul!(tmp, nav.P, transpose(Φ))
    mul!(nav.P, Φ, tmp)
    nav.P .+= Q
    return
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
function kalmanUpdate!(nav::NavStateEKF, t, y, h)
    if nav.iter > 0
        return kalmanUpdateIter!(nav, t, y, h, nav.iter)  # This is an IEKF
    end

    δy, δz, isRejected = kalmanUpdateError!(nav, t, y, h)
    nav.x .+= nav.δx
    nav.δx .= 0.0       # reset error state

    return δy, δz, isRejected
end

"""
    kalmanUpdateError!(nav, t, y, h)

Update error state of the Kalman filter using the input measurement.

Inputs include the measurement time ```t```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x)```. This function is only applicable
to EKF and UDEKF.
"""
@views function kalmanUpdateError!(nav::NavStateEKF, t, y, h)
    # Predict measurement, and compute noise covariance matrix and jacobian
    ŷ, R, H = h(t, nav.x)

    # Allocate innovation and normalized innovation
    δy = zero(y); δz = zero(y)

    # Perform kalman update
    isRejected = kalmanUpdateError!(nav, y, ŷ, R, H, δy, δz)

    # Return results
    return δy, δz, isRejected
end

@views function kalmanUpdateError!(nav::NavStateEKF, y, ŷ, R, H,
        δy = zero(y),                                                       # Save allocations
        δz = zero(y),                                                       # Save allocations
        Pxy = Matrix{eltype(nav.P)}(undef, nav.nδ, length(y)),              # Save allocations
        Pyy = Matrix{eltype(nav.P)}(undef, size(R)),                        # Save allocations
        PyyK = Matrix{eltype(nav.P)}(undef, length(y), nav.ns),             # Save allocations
    )

    isRejected = false

    # Estimated measurement and jacobians
    mul!(Pxy, nav.P, H')    # Pxy = P*Hᵀ
    mul!(Pyy, H, Pxy)       # Pyy = H*P*Hᵀ + R
    Pyy .+= R

    # Measurement editing
    # δy := y - (ŷ + H*δx)
    mul!(δy, H, nav.δx)
    @inbounds for i in eachindex(y)
        # Check negative covariance (numerical issue)
        if Pyy[i, i] < 0
            isRejected = true
            break
        end

        # Innovation and normalized innovation
        δy[i] = y[i] - ŷ[i] - δy[i]     # Fix innovation definition wrt mul!()
        δz[i] = δy[i]/sqrt(Pyy[i, i])

        # Check rejection threshold
        if abs(δz[i]) > nav.σᵣ
            isRejected = true
            break
        end
    end

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        Ks = Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
        mul!(nav.xs, Ks, δy)
        nav.δx[1:nav.ns] .+= nav.xs

        # Covariance update (non-optimal gain with consider states)
        mul!(PyyK, Pyy, transpose(Ks))
        mul!(nav.KPyyK, Ks, PyyK)
        nav.P[1:nav.ns, 1:nav.ns] .-= nav.KPyyK         # Ks*Pyy*Ks'
        mul!(nav.KPyx, Ks, transpose(Pxy[nav.ns+1:nav.nδ, :]))
        nav.P[1:nav.ns, nav.ns+1:nav.nδ] .-= nav.KPyx
        @inbounds for ir in nav.ns+1:nav.nδ, ic in 1:nav.ns
            nav.P[ir, ic] = nav.P[ic, ir]       # Make it symmmetric
        end
    end

    return isRejected
end

# Scalar measurement update for EKF
# The following function can be directly used when R is a diagonal matrix
@views function kalmanUpdateErrorScalar!(nav::NavStateEKF, t, y, h)
    ŷ, R, H = h(t, nav.x)
    δy = zero(y)
    δz = zero(y)

    isRejected = kalmanUpdateErrorScalar!(nav, y, ŷ, R, H, δy, δz)

    return δy, δz, isRejected
end

# The following function can be directly used when R is a diagonal matrix
@views function kalmanUpdateErrorScalar!(nav::NavStateEKF, y, ŷ, R, H,
        δy = zero(y),                                                       # Save allocations
        δz = zero(y),                                                       # Save allocations
    )
    isRejected = false
    Ks = nav.xs
    Pxy = nav.pxy

    @inbounds for i in eachindex(y)
        # Estimated measurement and jacobians
        mul!(Pxy, nav.P, H[i, :])
        Pyy = dot(H[i, :], Pxy) + R[i, i]

        if Pyy < 0.0
            isRejected = true
            break
        end

        # Measurement editing
        δy[i] = y[i] - (ŷ[i] + dot(H[i, :], nav.δx))
        δz[i] = δy[i]/sqrt(Pyy)                     # Normalized innovation
        isRejected = abs(δz[i]) > nav.σᵣ            # σ rejection threshold

        # Update error state and covariance matrix
        if !isRejected
            # Error state update
            @inbounds for j in 1:nav.ns
                Ks[j] = Pxy[j]/Pyy    # Kalman Gain
                nav.δx[j] += Ks[j]*δy[i]
            end

            # Covariance update (non-optimal gain with consider states)
            # P[1:ns, 1:ns] -= Pyy * Ks * Ks'
            mul!(nav.KPyyK, Ks, transpose(Ks))       # KPyyK = Ks * Ks'
            rmul!(nav.KPyyK, Pyy)                    # KPyyK *= Pyy
            nav.P[1:nav.ns, 1:nav.ns] .-= nav.KPyyK  # In-place subtraction

            # P[1:ns, ns+1:nδ] -= Ks * Pxy[ns+1:nδ, :]'
            mul!(nav.KPyx, Ks, transpose(Pxy[nav.ns+1:nav.nδ]))       # KPyx = Ks*Pxyᵀ
            nav.P[1:nav.ns, nav.ns+1:nav.nδ] .-= nav.KPyx             # In-place subtraction

            # nav.P[nav.ns+1:nav.nδ, 1:nav.ns] .= transpose(nav.P[1:nav.ns, nav.ns+1:nav.nδ])
            @inbounds for ir in nav.ns+1:nav.nδ, ic in 1:nav.ns
                nav.P[ir, ic] = nav.P[ic, ir]       # Make it symmmetric
            end
        end
    end

    return isRejected
end

# This update routine implements an IKEF
@views function kalmanUpdateIter!(nav::NavStateEKF, t, y, h, iter)
    # Estimated measurement and jacobians
    ŷ, R, H = h(t, nav.x)
    Pxy = nav.P*H'
    Pyy = H*Pxy + R

    # Measurement editing
    δy = y - ŷ
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs, δz) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    Ks = zeros(nav.ns, length(y))
    if !isRejected
        xIter = copy(nav.x)

        # Start iterations
        @inbounds for i in 1:iter
            if i > 1
                ŷ, R, H = h(t, xIter)
                mul!(Pxy, nav.P, H')
                mul!(Pyy, H, Pxy)
                Pyy .+= R
            end

            # State update
            Ks .= Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
            xIter[1:nav.ns] .= nav.x[1:nav.ns] + Ks*(y - ŷ - H*(nav.x - xIter))
        end

        # Update state
        nav.x .= xIter

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns, 1:nav.ns] .-= Ks*Pyy*Ks'
        nav.P[1:nav.ns, nav.ns+1:nav.nδ] .-= Ks*Pxy[nav.ns+1:nav.nδ, :]'
        @inbounds for ir in nav.ns+1:nav.nδ, ic in 1:nav.ns
            nav.P[ir, ic] = nav.P[ic, ir]       # Make it symmmetric
        end
    end

    return δy, δz, isRejected
end
