mutable struct NavStateUKF
    t::Float64              # Time corresponding to the estimated state
    x̂::Vector{Float64}      # Full estimated state, x̂[t]
    P::Matrix{Float64}      # Covariance matrix P[t]
    ns::UInt8               # Number of solve for states
    σᵣ::UInt8               # Outlier rejection threshold
    γ::Float64              # UKF parameters
    Wm::Vector{Float64}     # UKF parameters
    Wc::Vector{Float64}     # UKF parameters
    L::UInt8                    # Length of state vector
    X̂::Vector{Vector{Float64}}  # Sigma point states
end

"""
    NavStateUKF(t, x̂, P)

Build UKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateUKF(t, x̂, P; α=1e-3, β=2.0, κ=0.0)
    L = size(x̂, 1)
    γ, Wm, Wc = UKFweights(L, α, β, κ)

    return NavStateUKF(t, x̂, P, L, 6, γ, Wm, Wc, L, [zeros(L) for _ in 1:2L+1])
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
    nav.X̂[1] = nav.x̂
    nav.X̂[2:nav.L+1] = [nav.x̂ + nav.γ*S[i,:] for i in 1:nav.L]
    nav.X̂[nav.L+2:end] = [nav.x̂ - nav.γ*S[i,:] for i in 1:nav.L]
end

function kalmanPropagate!(nav::NavStateUKF, Δt, f, Jf, Q; nSteps=1)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points
    nav.X̂ = odeCore.(Ref(nav.t), nav.X̂, Ref(Δt), Ref(f); nSteps=nSteps)
    nav.t = nav.t + Δt

    # Compute mean state
    nav.x̂ = sum(nav.Wm.*nav.X̂)

    # Compute covariance estimate
    nav.P = copy(Q)
    for i in 1:2*nav.L+1
        δX = nav.X̂[i] - nav.x̂
        nav.P = nav.P + nav.Wc[i].*δX*transpose(δX)
    end
end

function kalmanPropagate!(nav::NavStateUKF, Δt, f, Q; nSteps=1)
    kalmanPropagate!(nav, Δt, f, nothing, Q, nSteps=nSteps)
end

function kalmanUpdate!(nav::NavStateUKF, ty, y, h)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    out = h.(Ref(ty),nav.X̂)
    Ŷ = getindex.(out,1)
    ŷ = sum(nav.Wm.*Ŷ)

    # Compute sigma statistics
    Pxy = zeros(nav.L,length(ŷ))
    Pyy = getindex.(out,2)[1]   # R
    for i in 1:2*nav.L+1
        δY = Ŷ[i] - ŷ
        δX = nav.X̂[i] - nav.x̂
        Pyy = Pyy + nav.Wc[i].*δY*transpose(δY)
        Pxy = Pxy + nav.Wc[i].*δX*transpose(δY)
    end

    # Measurement editing
    δy = y - ŷ
    δz = δy./sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if ~isRejected
        # Error state update
        Ks = Pxy[1:nav.ns,:]/Pyy     # Kalman Gain
        nav.x̂[1:nav.ns] = nav.x̂[1:nav.ns] + Ks*δy

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns,:] = nav.P[1:nav.ns,:] - Ks*[Pyy*transpose(Ks) transpose(Pxy[nav.ns+1:end,:])]
        nav.P[nav.ns+1:end,1:nav.ns] = transpose(nav.P[1:nav.ns,nav.ns+1:end])
    end

    return δy, δz, isRejected
end
