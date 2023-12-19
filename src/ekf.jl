mutable struct NavState
    t::Float64              # Time corresponding to the estimated state
    x̂::Vector{Float64}      # Full estimated state, x̂[t]
    P::Matrix{Float64}      # Covariance Matrix, P[t]
    δx::Vector{Float64}     # Error state, δx[t]
    ns::Int64               # Number of solve for (error) states
    σᵣ::Int64               # Outlier rejection threshold
    nδ::Int64               # Number of error states
end

"""
    NavState(t, x̂, P)

Build EKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavState(t, x̂, P)
    nδ = size(P, 1)
    return NavState(t, x̂, P, zeros(nδ), nδ, 6, nδ)
end

"""
    getCov(nav)

Get navigation covariance matrix ``P``.
"""
function getCov(nav::NavState)
    return nav.P
end

# This function propagates the full navigation state from the current
# time to the current time plus Δt using a Runge-Kutta algorithm. It
# also computes the state transition matrix by numerical integration
# of the Jacobian of the dynamics.
@views function kalmanOde(t0, x0, Δt, f, Jf, nδ; nSteps=1)
    # Append state transition matrix to state vector
    nx = size(x0, 1)
    x = [x0; Matrix(1.0I, nδ, nδ)[:]]
    fun(t, x) = [f(t, x[1:nx]); Jf(t, x[1:nx])[:]]

    # Call Runge-Kutta
    x = odeCore(t0, x, Δt, fun; nSteps = nSteps)

    # Output results
    return t0 + Δt, x[1:nx], reshape(x[nx+1:end], nδ, nδ)  # t, x, Φ
end

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
function kalmanPropagate!(nav::NavState, Δt, f, Jf, Q; nSteps=1)
    nav.t, nav.x̂, Φ = kalmanOde(nav.t, nav.x̂, Δt, f, Jf, nav.nδ; nSteps=nSteps)
    nav.P = Φ*nav.P*transpose(Φ) + Q
end

"""
    kalmanPropagate!(nav, Δt, f, Q; nSteps = 1)

Propagate navigation state forward in time for ```Δt``` time units.

Inputs include the dynamics function ```ẋ = f(t, x)```, and equivalent discrete-time process noise
covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
number of RK4 steps to be performed when numerically integrating the system's
dynamics.
"""
function kalmanPropagate!(nav::NavState, Δt, f, Q; nSteps=1)
    Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
    kalmanPropagate!(nav, Δt, f, Jf, Q, nSteps = nSteps)
end

"""
    kalmanUpdateError!(nav, ty, y, h)

Update error state of the Kalman filter using the input measurement.

Inputs include the measurement time ```ty```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x̂)```. This function is only applicable
to EKF and UDEKF.
"""
function kalmanUpdateError!(nav::NavState, ty, y, h)
    # Estimated measurement and jacobians
    ŷ, R, H = h(ty, nav.x̂)
    Pxy = nav.P*transpose(H)
    Pyy = H*Pxy + R

    # Measurement editing
    δy = y - (ŷ + H*nav.δx)
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if ~isRejected
        # Error state update
        Ks = Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
        nav.δx[1:nav.ns] += Ks*δy

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns, :] -= Ks*[Pyy*transpose(Ks) transpose(Pxy[nav.ns+1:end, :])]
        nav.P[nav.ns+1:end, 1:nav.ns] = transpose(nav.P[1:nav.ns, nav.ns+1:end])
    end

    return δy, δz, isRejected
end

"""
    kalmanUpdate!(nav, ty, y, h)

Update state of the Kalman filter using the input measurement.

Inputs include the measurement time ```ty```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x̂)```. When using SRUKF or UKF, the
measurement function only needs to provide ```ŷ``` and ```R``` as outputs.
"""
function kalmanUpdate!(nav::NavState, ty, y, h)
    δy, δz, isRejected = kalmanUpdateError!(nav, ty, y, h)
    nav.x̂ += nav.δx
    resetErrorState!(nav)

    return δy, δz, isRejected
end
