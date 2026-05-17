mutable struct NavStateUKF{T<:AbstractVector{Float64},M<:AbstractMatrix{Float64}} <: AbstractNavState
    t::Float64                  # Time corresponding to the estimated state
    x::T                        # Full estimated state, x[t]
    P::M                        # Covariance matrix P[t]
    const ns::Int64             # Number of solve for states
    const γ::Float64            # UKF parameters
    const Wm::Vector{Float64}   # UKF parameters
    const Wc::Vector{Float64}   # UKF parameters
    const nx::Int64             # Length of state vector
    X::Vector{T}                # Sigma point states
    S::Matrix{Float64}          # Sqrt matrix (avoid allocations)
    odeCache::ODECache{T, M}
end

"""
    NavStateUKF(t, x, P)

Build UKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateUKF(t, x, P, ns=size(P, 1); α=1e-3, β=2.0, κ=0.0)
    L = length(x)
    γ, Wm, Wc = UKFweights(L, α, β, κ)
    odeCache = ODECache(x, P)
    return NavStateUKF(t, x, P, ns, γ, Wm, Wc, L, [zero(x) for _ in 1:(2L + 1)], zeros(ns, ns), odeCache)
end

@inline function getCov(nav::NavStateUKF)
    return nav.P
end

function UKFweights(L::Int, α=1e-3, β=2.0, κ=0.0)
    λ = α^2*(L + κ) - L
    γ = sqrt(L + λ)

    Wm = 0.5 / (L + λ) * ones(2*L+1)
    Wm[1] = λ / (L + λ)

    Wc = copy(Wm)
    Wc[1] += (1.0 - α^2 + β)

    return γ, Wm, Wc
end

function computeSigmaPoints!(nav::NavStateUKF)
    nx = nav.nx
    γ = nav.γ
    x = nav.x
    X = nav.X
    S = nav.S

    S .= nav.P
    S = cholesky!(Hermitian(S)).U    # S = sqrt(nav.P)
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

kalmanPropagate!(nav, Δt, f!, Jf!, p, Q; nSteps=1) = kalmanPropagate!(nav, Δt, f!, p, Q, nSteps=nSteps)

function kalmanPropagate!(nav::NavStateUKF, Δt, f!, p, Q; nSteps=1)
    # Extract from nav
    X = nav.X
    x = nav.x
    P = nav.P
    nx = nav.nx
    oc = nav.odeCache

    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points and compute mean state
    fill!(x, 0)
    @inbounds for i in eachindex(X)
        Xi = X[i]
        odeSolve!(Xi, nav.t, Δt, f!, p, oc; nSteps=nSteps)
        for j in 1:nx
            x[j] += nav.Wm[i] * Xi[j]
        end
    end
    nav.t += Δt

    # Compute covariance estimate
    @inbounds for i in eachindex(Q)
        P[i] = Q[i]
    end
    @inbounds for i in 1:(2*nx+1)
        δX = X[i]
        δX .-= x
        BLAS.ger!(nav.Wc[i], δX, δX, P)
    end
end

# h!(meas, x, p, t), shall fill meas.y, meas.H (if EKF), and meas.R
function kalmanUpdate!(nav::NavStateUKF, y, h!, meas::NavMeasurement=NavMeasurement(nav.nx, length(y)), p=nothing, t=nothing)
    # Init and extract variables
    ny = length(y)
    nx = nav.nx
    ns = nav.ns
    nX = 2*nx + 1
    x = nav.x
    X = nav.X
    P = nav.P
    δy = meas.δy
    δz = meas.δz
    Pxy = meas.Pxy
    Pyy = meas.Pyy
    K = meas.K
    Ŷ = meas.Y
    nReject = meas.nReject

    # Create sigma points
    computeSigmaPoints!(nav)

    # Compute mean estimated measurement
    @inbounds for i in 1:nX
        h!(meas, X[i], p, t)
        Ŷ[i] .= meas.y
    end
    ŷ = meas.y
    fill!(ŷ, 0)
    @inbounds for i in 1:nX
        Ŷi = Ŷ[i]
        for j in 1:ny
            ŷ[j] += nav.Wm[i] * Ŷi[j]
        end
    end

    # Compute sigma statistics
    Pyy .= meas.R
    fill!(Pxy, 0)
    @inbounds for i in 1:nX
        # Compute δY (this overwrites Ŷ[i] to reduce allocations)
        δY = Ŷ[i]
        for j in 1:ny
            δY[j] -= ŷ[j]
        end

        # Compute δY  (this overwrites X[i] to reduce allocations)
        δX = X[i]
        for j in 1:nx
            δX[j] -= x[j]
        end

        # BLAS rank-1 updates
        Wci = nav.Wc[i]
        BLAS.ger!(Wci, δY, δY, Pyy)
        BLAS.ger!(Wci, δX, δY, Pxy)
    end

    # Measurement editing
    @inbounds for i in eachindex(y)
        # Check negative covariance (numerical issue)
        Pyy[i, i] ≤ 0 && return true

        # Innovation and normalized innovation
        δy[i] = y[i] - ŷ[i]
        δz[i] = δy[i] / sqrt(Pyy[i, i])

        # Check rejection threshold
        abs(δz[i]) > nReject && return true
    end

    # Compute Kalman Gain
    K .= Pxy
    rdiv!(K, cholesky!(Hermitian(Pyy)))        # K = Pxy / Pyy, Caution: this modifies Pyy

    # Update error state and covariance matrix (non-optimal gain with consider states)
    @inbounds for r in 1:ns, j in 1:ny
        # P[1:ns, 1:ns] .-= Ks * Pyy * Ks' = -Pxy * Ks'
        # P[1:ns, (ns + 1):nδ] .-= Ks * Pyx
        pxy = Pxy[r, j]
        k = K[r, j]

        # Update state
        x[r] += k * δy[j]

        # Top left block: P[1:ns, 1:ns] (upper triangular only)
        for c in r:ns
            P[r, c] -= pxy * K[c, j]
        end

        # Top right block: P[1:ns, (ns + 1):nδ]
        for c in ns+1:nx
            P[r, c] -= k * Pxy[c, j]
        end
    end

    # Make covariance matrix symmetric
    # P[1:ns, 1:ns] (lower triangular only)
    @inbounds for r in 2:ns, c in 1:r-1
        P[r, c] = P[c, r]
    end
    # P[ns+1:nδ, 1:ns]
    @inbounds for r in (ns + 1):nx, c in 1:ns
        P[r, c] = P[c, r]
    end

    return false
end

# y = f(x, p)
function unscentedTransform(f, x, Pxx, p=nothing)
    nav = NavState(0.0, x, Pxx; type=:UKF)

    # Extract from nav
    y = f(x, p)
    X = nav.X
    ny = length(y)
    Pyy = zeros(ny, ny)
    Y = [zero(y) for _ in eachindex(X)]

    # Create sigma points
    computeSigmaPoints!(nav)

    # Propagate sigma points and compute mean state
    fill!(y, 0)
    @inbounds for i in eachindex(X)
        Yi = Y[i]
        Yi .= f(X[i], p)
        for j in eachindex(Yi)
            y[j] += nav.Wm[i] * Yi[j]
        end
    end

    # Compute covariance estimate
    fill!(Pyy, 0)
    @inbounds for i in 1:(2 * nav.nx + 1)
        δY = Y[i]
        δY .-= y
        BLAS.ger!(nav.Wc[i], δY, δY, Pyy)
    end

    return y, Pyy
end
