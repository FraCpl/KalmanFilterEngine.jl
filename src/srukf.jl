mutable struct NavStateSRUKF <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x̂::Vector{Float64}      # Full estimated state, x̂[t]
    S::Matrix{Float64}      # Cholesky decomposition of covariance matrix S[t]
    ns::Int64               # Number of solve for states
    σᵣ::Int64               # Outlier rejection threshold
    γ::Float64              # UKF parameters
    Wm::Vector{Float64}     # UKF parameters
    Wc::Vector{Float64}     # UKF parameters
    L::Int64                    # Length of state vector
    X̂::Vector{Vector{Float64}}  # Sigma point states
end

"""
    NavStateSRUKF(t, x̂, P)

Build SRUKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateSRUKF(t, x̂, P; α=1e-3, β=2.0, κ=0.0)
    S = cholesky(P).U
    L = size(x̂, 1)
    γ, Wm, Wc = UKFweights(L, α, β, κ)

    return NavStateSRUKF(t, x̂, S, L, 6, γ, Wm, Wc, L, [zeros(L) for _ in 1:2L+1])
end

getCov(nav::NavStateSRUKF) = nav.S'*nav.S

function computeSigmaPoints!(nav::NavStateSRUKF)
    nav.X̂[1] = nav.x̂
    nav.X̂[2:nav.L+1] = [nav.x̂ + nav.γ*nav.S[i,:] for i in 1:nav.L]
    nav.X̂[nav.L+2:end] = [nav.x̂ - nav.γ*nav.S[i,:] for i in 1:nav.L]
end

function kalmanPropagate!(nav::NavStateSRUKF, Δt, f, Jf, Q; nSteps=1)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points
    nav.X̂ = odeCore.(nav.t, nav.X̂, Δt, f; nSteps=nSteps)
    nav.t = nav.t + Δt

    # Compute mean state
    nav.x̂ = sum(nav.Wm.*nav.X̂)

    # Calculate covariance estimate
    M = zeros(nav.L,2*nav.L)
    wc = sqrt(nav.Wc[2])
    for i in 1:2*nav.L
        M[:,i] = wc*(nav.X̂[i+1] - nav.x̂)
    end
    nav.S = qr([M sqrt(Q)]').R

    δX1 = sqrt(abs(nav.Wc[1]))*(nav.X̂[1] - nav.x̂)
    cholupdate!(nav.S, δX1, sign(nav.Wc[1]))
end

function kalmanPropagate!(nav::NavStateSRUKF, Δt, f, Q; nSteps=1)
    kalmanPropagate!(nav, Δt, f, nothing, Q, nSteps = nSteps)
end

function kalmanUpdate!(nav::NavStateSRUKF, t, y, h)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    out = h.(t, nav.X̂)
    Ŷ = getindex.(out, 1)
    ŷ = sum(nav.Wm.*Ŷ)
    sqrtR = sqrt(getindex.(out, 2)[1])
    ny = length(y)

    # Compute sigma statistics
    M = zeros(ny, 2nav.L)
    wc = sqrt(nav.Wc[2])
    for i in 1:2*nav.L
        M[:,i] = wc*(Ŷ[i+1] - ŷ)
    end
    Syy = qr([M sqrtR]').R
    δY1 = sqrt(abs(nav.Wc[1]))*(Ŷ[1] - ŷ)
    cholupdate!(Syy, δY1, sign(nav.Wc[1]))

    Pxy = zeros(nav.L,ny)
    for i = 1:2*nav.L+1
        Pxy = Pxy + nav.Wc[i].*(nav.X̂[i] - nav.x̂)*(Ŷ[i] - ŷ)'
    end

    # Measurement editing
    δy = y - ŷ
    δz = δy./sqrt.(diag(Syy'*Syy))              # Normalized innovation
    isRejected = maximum(abs.(δz)) > nav.σᵣ     # σ rejection threshold

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        K = (Pxy/Syy)/Syy'                  # Kalman Gain
        K[nav.ns+1:end, :] .= 0.0            # Consider states
        nav.x̂[1:nav.ns] += K[1:nav.ns, :]*δy

        U = K*Syy'
        for i in 1:ny
            cholupdate!(nav.S, U[:, i], -1.0)
        end
    end

    return δy, δz, isRejected
end

# https://math.stackexchange.com/questions/4318420/how-does-cholupdate-work
# https://en.wikipedia.org/wiki/Cholesky_decomposition
# Caution: This modifies both S and x!
function cholupdate!(S, x, signx=1.0)
    n = length(x)
    for k in 1:n
        r = sqrt(S[k, k]^2 + signx*x[k]^2)
        c = r/S[k, k]
        s = x[k]/S[k, k]
        S[k, k] = r
        if k < n
            S[k, k+1:n] = (S[k, k+1:n] + signx*s*x[k+1:n])/c
            x[k+1:n] = c*x[k+1:n] - s*S[k, k+1:n]
        end
    end
end
