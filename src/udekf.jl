mutable struct NavStateUD <: AbstractNavState
    t::Float64              # Time corresponding to the estimated state
    x#::Vector{Float64}      # Full estimated state, x[t]
    U#::Matrix{Float64}      # Covariance Matrix UD, U[t]
    D#::Vector{Float64}      # Covariance Matrix UD, D[t]
    δx#::Vector{Float64}     # Error state, δx[t]
    ns::Int64               # Number of solve for (error) states
    σᵣ::Int64               # Outlier rejection threshold
    nδ::Int64               # Number of error states
end

"""
    NavStateUD(t, x, P)

Build UDEKF navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavStateUD(t, x, P)
    U, D = UD(P)
    nδ = size(U, 1)
    return NavStateUD(t, x, U, D, 0*P[:, 1], nδ, 6, nδ)
end

getCov(nav::NavStateUD) = nav.U*diagm(nav.D)*nav.U'

@views function UD(P)
    n = size(P, 1)
    U = Matrix(1.0I, n, n)
    D = zeros(n)
    D[end] = P[end]
    if abs(P[end]) > 1e-9
        U[:, end] = P[:, end]./P[end]
    end
    for j in n-1:-1:1
        D[j] = P[j, j] - sum(D[j+1:n].*U[j, j+1:n].^2)
        if D[j] > 0.0
            for i in j-1:-1:1
                U[i, j] = (P[i, j] - sum(D[j+1:n].*U[i, j+1:n].*U[j, j+1:n]))/D[j]
            end
        end
    end

    return U, D
end

# Agee-Turner rank one update algorithm
# Utilde*Dtilde*Utilde' = U*D*U' + c.*x*x'
#
# D must be provided as a vector.
# c is a positive scalar.
# x is a vector.
@views function ageeTurnerUpdate(U, D, c, x)
    if c > 0
        xx = copy(x)
        n = length(D)
        Ũ = Matrix(1.0I, n, n)
        D̃ = zeros(n)
        for j in n:-1:2
            D̃[j] = D[j] + c*xx[j]^2
            b = c/D̃[j]
            v = b*xx[j]
            for i in 1:j-1
                xx[i] = xx[i] - U[i, j]*xx[j]
                Ũ[i, j] = U[i, j] + xx[i]*v
            end
            c = b*D[j]
        end
        D̃[1] = D[1] + c*xx[1]^2

        return Ũ, D̃
    end

    return U, D
end

# Carlson Rank-One Update
#
# This function computes K and performs the following operation
# U*D*U' = Ū*D̄*Ū - K*H*Ū*D̄*Ū'
#
# It also provides
# alpha = H*Ū*D̄*Ū'*H' + R
@views function carlsonUpdate(Ū, D̄, H, R)
    n = length(D̄)
    K̄ = zeros(n)
    D = zeros(n)
    U = Matrix(1.0I, n, n)
    f = zeros(n)

    f[1] = H[1]
    for i in 2:n
        f[i] = (Ū[1:i, i:i]'*H[1:i])[1]   # f = Ū'*H';
    end
    v = D̄.*f
    K̄[1] = v[1]
    αOld = R + v[1]*f[1]
    D[1] = R/αOld*D̄[1]

    α = αOld
    for i in 2:n
        α = αOld + v[i]*f[i]
        D[i] = αOld/α*D̄[i]
        U[:,i] = Ū[:,i] - f[i]/αOld*K̄
        K̄ = K̄ + v[i]*Ū[:,i]
        αOld = α
    end
    K = K̄./α     # Kopt

    return K, U, D, α
end

# Modified Weighted Gram-Schmidt Orthogonalisation Algorithm
# Ū*D̄*Ū' = Φ*U*D*U'*Φ' + Φw*Q*Φw'
#
# Q must be diagonal (and provided as a vector)
# D must be provided as a vector
@views function modGramSchmidt(Φ, U, D, Φw, Q)
    n = size(Φ, 1)
    Ū = Matrix(1.0I, n, n)
    D̄ = zeros(n)
    D̃ = [D; Q]

    b = [Φ*U Φw]'
    for j in n:-1:2
        f = D̃.*b[:, j]
        D̄[j] = b[:, j]'*f
        f = f./D̄[j]
        for i in 1:j-1
            Ū[i,j] = b[:, i]'*f
            b[:,i] = b[:, i] - Ū[i, j]*b[:, j]
        end
    end
    D̄[1] = b[:, 1]'*(D̃.*b[:, 1])

    return Ū, D̄
end

# Modified Weighted Gram-Schmidt Orthogonalisation Algorithm
# Ubar*Dbar*Ubar' = Phi*U*D*U'*Phi'
#
# D must be provided as a vector
@views function modGramSchmidtReduced(Φ, U, D)
    n = size(Φ,1)
    Ū = Matrix(1.0I, n, n)
    D̄ = zeros(n)

    b = (Φ*U)'
    for j = n:-1:2
        f = D.*b[:, j]
        D̄[j] = b[:, j]'*f
        if D̄[j] > 0              # CHECK THIS IF, there was none before
            f = f./D̄[j]
        end
        for i = 1:j-1
            Ū[i, j] = b[:, i]'*f;
            b[:, i] = b[:, i] - Ū[i, j]*b[:, j];
        end
    end
    D̄[1] = b[:, 1]'*(D.*b[:, 1]);

    return Ū, D̄
end

function kalmanPropagate!(nav::NavStateUD, Δt, f, Jf, Q; nSteps=1)
    nav.t, nav.x, Φ = kalmanOde(nav.t, nav.x, Δt, f, Jf, nav.nδ; nSteps=nSteps)
    nav.U, nav.D = UDpropagate(nav.U, nav.D, Φ, Q, size(nav.x, 1))
end

function kalmanPropagate!(nav::NavStateUD, Δt, f, Q; nSteps=1)
    Jf(t, x) = ForwardDiff.jacobian(x -> f(t, x), x)
    kalmanPropagate!(nav, Δt, f, Jf, Q, nSteps=nSteps)
end

# UD covariance propagation, taking into account non-correlated states
# (e.g., measurement biases, ecrvs)
# Ū*D̄*Ū' = Φ*U*D*U'*Φ' + Q
#
# nc: number of fully correlated states
# Φ = [Φxx Φxp; 0 Φpp]
# Q = [Qxx 0; 0 diag(Qpp)]
# Φpp of non correlated states must be diagonal
# Q of non correlated states must be diagonal (Qpp diagonal and Qxp = 0)
function UDpropagate(U, D, Φ, Q, nc)
    nδ = size(Φ, 1)
    np = nδ - nc       # Number of parameters (i.e., non correlated states)

    # Solve first sub-problem
    Φxx = Φ[1:nc, 1:nc]
    Uxx = U[1:nc, 1:nc]
    Dxx = D[1:nc]
    Qxx = Q[1:nc, 1:nc]
    if any(Qxx .!= 0.0)
        if isdiag(Qxx)
            # CASE 1: Diagonal Qxx
            # Noiseless propagation
            Ũxx, D̃xx = modGramSchmidtReduced(Φxx, Uxx, Dxx)

            # Add single noise components with ageeTurnerUpdate
            for i in 1:nc
                if Qxx[i, i] > 0
                    x = zeros(nc); x[i] = 1.0
                    Ũxx, D̃xx = ageeTurnerUpdate(Ũxx, D̃xx, Qxx[i, i], x)
                end
            end
        else
            # CASE 2: Generic, non-diagonal Qxx
            # Diagonalize
            ΦWxx, Qxx = UD(Qxx)

            # Apply complete Gram Schmidt update
            Ũxx, D̃xx = modGramSchmidt(Φxx, Uxx, Dxx, ΦWxx, Qxx)
        end
    else
        # CASE 3: Qxx = 0
        # Noiseless propagation
        Ũxx, D̃xx = modGramSchmidtReduced(Φxx, Uxx, Dxx)
    end

    # Solve second sub-problem
    if np == 0
        return Ũxx, D̃xx
    end

    Ũ = [Ũxx Φ[1:nc, :]*U[:, nc+1:nδ]; zeros(np,nc) U[nc+1:nδ, nc+1:nδ]]    # Eq. (7.27) (7.29)
    D̃ = [D̃xx; D[nc+1:nδ]]                                                  # Eq. (7.28)

    # Reference: C. L. Thornton, Triangular Covariance Factorizations for Kalman Filtering, 1976, page 61
    Qpp = diag(Q[nc+1:nδ, nc+1:nδ])
    M = diag(Φ[nc+1:nδ, nc+1:nδ])
    Ū = copy(Ũ); D̄ = copy(D̃)
    for k in 1:np
        na = nc + k - 1
        D̄[na+1] = M[k]^2*D̃[na+1] + Qpp[k]                   # d_up_b, Eq. (7.38)
        α = M[k]*D̃[na+1]/D̄[na+1]                            # [Default]
        Ū[1:na, na+1] = α.*Ũ[1:na, na+1]                      # U_up_ab, Eq. (7.39)
        Ū[na+1, na+2:nδ] = M[k].*Ũ[na+1, na+2:nδ]  	      # U_up_bc, Eq. (7.37)
        if Qpp[k] > 0
            c = α*Qpp[k]/M[k]
            Ū[1:na, 1:na], D̄[1:na] = ageeTurnerUpdate(Ū[1:na, 1:na], D̄[1:na], c, Ũ[1:na, na+1])  # Eq. (7.40)
        end
    end

    return Ū, D̄
end

@views function kalmanUpdateError!(nav::NavStateUD, t, y, h)
    # Estimated measurement and jacobians
    ŷ, R, H = h(t, nav.x)

    # Decorrelate measurement if R is not diagonal
    if ~isdiag(R)
        y, ŷ, R, H = decorrelateMeas(y, ŷ, R, H)
    end

    ny = length(y)
    δy = zeros(ny); δz = zeros(ny)
    nx = length(nav.D)
    isRejected = false

    for i in 1:ny
        W1 = H[i:i,:]*nav.U
        Ph = W1*diagm(nav.D)*W1'

        # Measurement Editing (innovation check)
        δy[i] = y[i] - ŷ[i] - H[i, :]'*nav.δx
        δz[i] = δy[i]/sqrt(Ph[1] + R[i, i])
        isRejected = abs(δz[i]) > nav.σᵣ

        # Update the state and covariance estimates for non-optimal K
        # (to be used with consider and underweighted gain instead of the optimal formula: P = P - K*Pyy*K')
        if !isRejected
            K, nav.U, nav.D, α = carlsonUpdate(nav.U, nav.D, H[i, :], R[i, i])  # Kopt

            # Perform Agee-Turner rank-one update to account for consider states
            if nx > nav.ns
                nav.U, nav.D = ageeTurnerUpdate(nav.U,nav.D,α,[zeros(nav.ns); K[nav.ns+1:nav.nδ]]);
            end

            nav.δx[1:nav.ns] += K[1:nav.ns]*δy[i]
        end
    end

    return δy, δz, isRejected
end

function kalmanUpdate!(nav::NavStateUD, t, y, h)
    δy, δz, isRejected = kalmanUpdateError!(nav, t, y, h)
    nav.x .+= nav.δx
    resetErrorState!(nav)
    return δy, δz, isRejected
end
