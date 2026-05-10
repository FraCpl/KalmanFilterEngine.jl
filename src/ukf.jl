mutable struct NavStateUKF{T<:AbstractVector{Float64},M<:AbstractMatrix{Float64}} <: AbstractNavState
    t::Float64                  # Time corresponding to the estimated state
    x::T                        # Full estimated state, x[t]
    P::M                        # Covariance matrix P[t]
    const ns::Int64             # Number of solve for states
    const σᵣ::Int64             # Outlier rejection threshold
    const γ::Float64            # UKF parameters
    const Wm::Vector{Float64}   # UKF parameters
    const Wc::Vector{Float64}   # UKF parameters
    const L::Int64              # Length of state vector
    X::Vector{T}                # Sigma point states
    odeCache::ODECache
end

"""
    NavStateUKF(t, x, P)

Build UKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateUKF(t, x, P, ns=size(P, 1); α=1e-3, β=2.0, κ=0.0)
    L = size(x, 1)
    γ, Wm, Wc = UKFweights(L, α, β, κ)
    odeCache = ODECache(x, P)
    return NavStateUKF(t, x, P, ns, 6, γ, Wm, Wc, L, [zero(x) for _ in 1:(2L + 1)], odeCache)
end

@inline function getCov(nav::NavStateUKF)
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

@inline function computeSigmaPoints!(nav::NavStateUKF)
    S = sqrt(nav.P)
    nav.X[1] .= nav.x
    @inbounds for i in 1:nav.L, j in 1:nav.L
        nav.X[i + 1][j] = nav.x[j] + nav.γ*S[i, j]
        nav.X[i + 1 + nav.L][j] = nav.x[j] - nav.γ*S[i, j]
    end
end

function kalmanPropagate!(nav::NavStateUKF, Δt, f!, Jf!, p, Q; nSteps=1)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points and compute mean state
    fill!(nav.x, 0)
    @inbounds for i in eachindex(nav.X)
        odeSolve!(nav.X[i], nav.t, Δt, f!, p, nav.odeCache; nSteps=nSteps)
        nav.x .+= nav.X[i] .* nav.Wm[i]
    end
    nav.t = nav.t + Δt

    # Compute covariance estimate
    @inbounds for i in eachindex(Q)
        nav.P[i] = Q[i]
    end
    @inbounds for i in 1:(2 * nav.L + 1)
        δX = nav.X[i] - nav.x
        nav.P .+= nav.Wc[i] .* δX*δX'
    end
end

@inline function kalmanPropagate!(nav::NavStateUKF, Δt, f, p, Q; nSteps=1)
    kalmanPropagate!(nav, Δt, f, nothing, p, Q, nSteps=nSteps)
end

@views function kalmanUpdate!(nav::NavStateUKF, t, y, h)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    out = h.(t, nav.X)
    Ŷ = getindex.(out, 1)
    ŷ = sum(nav.Wm .* Ŷ)

    # Compute sigma statistics
    Pxy = zeros(nav.L, length(ŷ))
    Pyy = getindex.(out, 2)[1]   # R
    @inbounds for i in 1:(2 * nav.L + 1)
        δY = Ŷ[i] - ŷ
        δX = nav.X[i] - nav.x
        Pyy .+= nav.Wc[i] .* δY*δY'
        Pxy .+= nav.Wc[i] .* δX*δY'
    end

    # Measurement editing
    δy = y - ŷ
    δz = δy ./ sqrt.(diag(Pyy))                   # Normalized innovation
    isRejected = maximum(abs, δz) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        Ks = Pxy[1:nav.ns, :]/Pyy     # Kalman Gain
        nav.x[1:nav.ns] .+= Ks*δy

        # Covariance update (non-optimal gain with consider states)
        nav.P[1:nav.ns, :] .-= Ks*[Pyy*Ks' Pxy[(nav.ns + 1):nav.L, :]']
        nav.P[(nav.ns + 1):nav.L, 1:nav.ns] = nav.P[1:nav.ns, (nav.ns + 1):nav.L]'
    end

    return δy, δz, isRejected
end
