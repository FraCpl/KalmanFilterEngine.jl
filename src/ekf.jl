mutable struct NavStateEKF <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x#::Vector{Float64}      # Full estimated state, x[t]
    P#::Matrix{Float64}      # Covariance Matrix, P[t]
    δx#::Vector{Float64}     # Error state, δx[t]
    ns::Int64               # Number of solve for (error) states
    σᵣ::Int64               # Outlier rejection threshold
    nδ::Int64               # Number of error states
    iter::Int64             # Number of iterations for IEKF
end

"""
    NavStateEKF(t, x, P)

Build EKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateEKF(t, x, P; iter=0)
    nδ = size(P, 1)
    return NavStateEKF(t, x, P, 0*P[:, 1], nδ, 6, nδ, iter)     # we do 0*P[:, 1] for compatibilty with ComponentArrays
end

"""
    getCov(nav)

Get navigation covariance matrix ``P``.
"""
getCov(nav::NavStateEKF) = nav.P

# This function propagates the full navigation state from the current
# time to the current time plus Δt using a Runge-Kutta algorithm. It
# also computes the state transition matrix by numerical integration
# of the Jacobian of the dynamics.
@views function kalmanOde(t0, x0, Δt, f, Jf, nδ; nSteps=1)
    x, Φ = odeCore(t0, x0, Matrix(1.0I, nδ, nδ), Δt, f, Jf; nSteps=nSteps)
    return t0 + Δt, x, Φ
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
function kalmanPropagate!(nav::NavStateEKF, Δt, f, Jf, Q; nSteps=1)
    nav.t, nav.x, Φ = kalmanOde(nav.t, nav.x, Δt, f, Jf, nav.nδ; nSteps=nSteps)
    nav.P = Φ*nav.P*Φ' + Q
end

"""
    kalmanPropagate!(nav, Δt, f, Q; nSteps = 1)

Propagate navigation state forward in time for ```Δt``` time units.

Inputs include the dynamics function ```ẋ = f(t, x)```, and equivalent discrete-time process noise
covariance matrix ```Q```. The optional keyword argument ```nSteps``` indicates the
number of RK4 steps to be performed when numerically integrating the system's
dynamics.
"""
function kalmanPropagate!(nav::NavStateEKF, Δt, f, Q; nSteps=1)
    Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
    kalmanPropagate!(nav, Δt, f, Jf, Q, nSteps = nSteps)
end

"""
    kalmanUpdateError!(nav, t, y, h)

Update error state of the Kalman filter using the input measurement.

Inputs include the measurement time ```t```, measurement ```y```,
measurement equation function ```ŷ, R, H = h(t, x)```. This function is only applicable
to EKF and UDEKF.
"""
function kalmanUpdateError!(nav::NavStateEKF, t, y, h)
    # Estimated measurement and jacobians
    ŷ, R, H = h(t, nav.x)
    Pxy = nav.P*H'
    Pyy = H*Pxy + R

    # Measurement editing
    δy = y - (ŷ + H*nav.δx)
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        Ks = Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
        nav.δx[1:nav.ns] += Ks*δy

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns, :] -= Ks*[Pyy*Ks' Pxy[nav.ns+1:end, :]']
        nav.P[nav.ns+1:end, 1:nav.ns] = nav.P[1:nav.ns, nav.ns+1:end]'
    end

    return δy, δz, isRejected
end

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
    resetErrorState!(nav)

    return δy, δz, isRejected
end

# This update routine implements an IKEF
function kalmanUpdateIter!(nav::NavStateEKF, t, y, h, iter)
    # Estimated measurement and jacobians
    ŷ, R, H = h(t, nav.x)
    Pxy = nav.P*H'
    Pyy = H*Pxy + R

    # Measurement editing
    δy = y - ŷ
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    Ks = zeros(nav.ns, length(y))
    if !isRejected
        xIter = copy(nav.x)

        # Start iterations
        for i in 1:iter
            if i > 1
                ŷ, R, H = h(t, xIter)
                Pxy .= nav.P*H'
                Pyy .= H*Pxy + R
            end

            # State update
            Ks .= Pxy[1:nav.ns, :]/Pyy    # Kalman Gain
            xIter[1:nav.ns] .= nav.x[1:nav.ns] + Ks*(y - ŷ - H*(nav.x - xIter))
        end

        # Update state
        nav.x .= xIter

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns, :] -= Ks*[Pyy*Ks' Pxy[nav.ns+1:end, :]']
        nav.P[nav.ns+1:end, 1:nav.ns] = nav.P[1:nav.ns, nav.ns+1:end]'
    end

    return δy, δz, isRejected
end
