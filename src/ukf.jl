mutable struct NavStateUKF <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x#::Vector{Float64}      # Full estimated state, x[t]
    P#::Matrix{Float64}      # Covariance matrix P[t]
    ns::Int64               # Number of solve for states
    σᵣ::Int64               # Outlier rejection threshold
    γ::Float64              # UKF parameters
    Wm::Vector{Float64}     # UKF parameters
    Wc::Vector{Float64}     # UKF parameters
    L::Int64                # Length of state vector
    X#::Vector{Vector{Float64}}  # Sigma point states
end

"""
    NavStateUKF(t, x, P)

Build UKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateUKF(t, x, P; α=1e-3, β=2.0, κ=0.0)
    L = size(x, 1)
    γ, Wm, Wc = UKFweights(L, α, β, κ)

    return NavStateUKF(t, x, P, L, 6, γ, Wm, Wc, L, [0*x for _ in 1:2L+1])
end

function getCov(nav::NavStateUKF)
    return nav.P
end

function UKFweights(L, α=1e-3, β=2.0, κ=0.0)
    λ = α^2*(L + κ) - L
    γ = sqrt(L + λ)

    Wm = 0.5/(L + λ)*ones(2*L+1)
    Wm[1] = λ/(L + λ)

    Wc = copy(Wm)
    Wc[1] += (1.0 - α^2 + β)

    return γ, Wm, Wc
end

function computeSigmaPoints!(nav::NavStateUKF)
    S = sqrt(nav.P)
    nav.X[1] = nav.x
    nav.X[2:nav.L+1] = [nav.x + nav.γ*S[i,:] for i in 1:nav.L]
    nav.X[nav.L+2:end] = [nav.x - nav.γ*S[i,:] for i in 1:nav.L]
end

function kalmanPropagate!(nav::NavStateUKF, Δt, f, Jf, Q; nSteps=1)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points
    nav.X = odeCore.(nav.t, nav.X, Δt, f; nSteps=nSteps)
    nav.t = nav.t + Δt

    # Compute mean state
    nav.x = sum(nav.Wm.*nav.X)

    # Compute covariance estimate
    nav.P = copy(Q)
    for i in 1:2*nav.L+1
        δX = nav.X[i] - nav.x
        nav.P += nav.Wc[i].*δX*δX'
    end
end

function kalmanPropagate!(nav::NavStateUKF, Δt, f, Q; nSteps=1)
    kalmanPropagate!(nav, Δt, f, nothing, Q, nSteps=nSteps)
end

@views function kalmanUpdate!(nav::NavStateUKF, t, y, h)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    out = h.(t, nav.X)
    Ŷ = getindex.(out, 1)
    ŷ = sum(nav.Wm.*Ŷ)

    # Compute sigma statistics
    Pxy = zeros(nav.L, length(ŷ))
    Pyy = getindex.(out, 2)[1]   # R
    for i in 1:2*nav.L+1
        δY = Ŷ[i] - ŷ
        δX = nav.X[i] - nav.x
        Pyy += nav.Wc[i].*δY*δY'
        Pxy += nav.Wc[i].*δX*δY'
    end

    # Measurement editing
    δy = y - ŷ
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        Ks = Pxy[1:nav.ns, :]/Pyy     # Kalman Gain
        nav.x[1:nav.ns] += Ks*δy

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns,:] -= Ks*[Pyy*Ks' Pxy[nav.ns+1:nav.L,:]']
        nav.P[nav.ns+1:nav.L,1:nav.ns] = nav.P[1:nav.ns,nav.ns+1:nav.L]'
    end

    return δy, δz, isRejected
end
