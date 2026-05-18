mutable struct NavStateSRUKF{T<:AbstractVector{Float64}} <: AbstractNavState
    t::Float64                  # Time corresponding to the estimated state
    x::T                        # Full estimated state, x[t]
    S::Matrix{Float64}          # Cholesky decomposition of covariance matrix S[t]
    const ns::Int64             # Number of solve for states
    const γ::Float64            # UKF parameters
    const Wm::Vector{Float64}   # UKF parameters
    const Wc::Vector{Float64}   # UKF parameters
    const nx::Int64             # Length of state vector
    X::Vector{T}                # Sigma point states
    odeCache::ODECache
end

"""
    NavStateSRUKF(t, x, P)

Build SRUKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateSRUKF(t, x, P, ns=size(P, 1); α=1e-3, β=2.0, κ=0.0)
    # @warning "WORK-IN-PROGRESS: might not work!"
    S = Matrix(cholesky(P).U)#.data
    nx = size(x, 1)
    γ, Wm, Wc = UKFweights(nx, α, β, κ)
    odeCache = ODECache(x, P)
    return NavStateSRUKF(t, x, S, ns, γ, Wm, Wc, nx, [zero(x) for _ in 1:(2L + 1)], odeCache)
end

getCov(nav::NavStateSRUKF) = nav.S'*nav.S

function computeSigmaPoints!(nav::NavStateSRUKF)
    nx = nav.nx
    γ = nav.γ
    x = nav.x
    X = nav.X
    S = nav.S
    X[1] .= x
    @inbounds for i in 1:nx
        X1 = X[i+1]
        X2 = X[i+1+nx]
        for j in 1:nx
            X1[j] = x[j] + γ*S[i, j]
            X2[j] = x[j] - γ*S[i, j]
        end
    end
end

@views function kalmanPropagate!(nav::NavStateSRUKF, Δt, f!, p, Q; nSteps=1)
    # Extract from nav
    X = nav.X; x = nav.x; P = nav.P

    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points
    fill!(x, 0)
    @inbounds for i in eachindex(X)
        Xi = X[i]
        odeSolve!(Xi, nav.t, Δt, f!, p, nav.odeCache; nSteps=nSteps)
        for j in eachindex(x)
            x[j] += Xi[j] * nav.Wm[i]
        end
    end
    nav.t += Δt

    # Calculate covariance estimate
    M = zeros(nav.nx, 2*nav.nx)
    wc = sqrt(nav.Wc[2])
    @inbounds for i in 1:(2 * nav.nx)
        M[:, i] .= wc*(nav.X[i + 1] - nav.x)
    end
    nav.S .= qr([M sqrt(Q)]').R

    δX1 = sqrt(abs(nav.Wc[1]))*(nav.X[1] - nav.x)
    cholupdate!(nav.S, δX1, sign(nav.Wc[1]))
end

@views function kalmanUpdate!(nav::NavStateSRUKF, t, y, h; nReject::Int=6)
    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    out = h.(t, nav.X)
    Ŷ = getindex.(out, 1)
    ŷ = sum(nav.Wm .* Ŷ)
    sqrtR = sqrt(getindex.(out, 2)[1])
    ny = length(y)

    # Compute sigma statistics
    M = zeros(ny, 2nav.nx)
    wc = sqrt(nav.Wc[2])
    @inbounds for i in 1:(2 * nav.nx)
        M[:, i] = wc*(Ŷ[i + 1] - ŷ)
    end
    Syy = qr([M sqrtR]').R
    δY1 = sqrt(abs(nav.Wc[1]))*(Ŷ[1] - ŷ)
    cholupdate!(Syy, δY1, sign(nav.Wc[1]))

    Pxy = zeros(nav.nx, ny)
    @inbounds for i in 1:(2 * nav.nx + 1)
        Pxy .+= nav.Wc[i] .* (nav.X[i] - nav.x)*(Ŷ[i] - ŷ)'
    end

    # Measurement editing
    δy = y - ŷ
    δz = δy ./ sqrt.(diag(Syy'*Syy))           # Normalized innovation
    isRejected = maximum(abs, δz) > nReject    # σ rejection threshold

    # Update error state and covariance matrix
    if !isRejected
        # Error state update
        K = (Pxy/Syy)/Syy'                  # Kalman Gain
        K[(nav.ns + 1):nav.nx, :] .= 0.0            # Consider states
        nav.x[1:nav.ns] .+= K[1:nav.ns, :]*δy

        U = K*Syy'
        @inbounds for i in 1:ny
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
    @inbounds for k in 1:n
        r = sqrt(S[k, k]^2 + signx * x[k] * x[k])
        c = r / S[k, k]
        s = x[k] / S[k, k]
        S[k, k] = r
        for j in (k + 1):n
            S[k, j] = (S[k, j] + signx * s * x[j]) / c
            x[j] = c * x[j] - s * S[k, j]
        end
    end
end
