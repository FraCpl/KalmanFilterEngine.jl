function odeCore(t0, x0, Δt, f; nSteps=1)
    # 3/8 Runge-Kutta Method
    # http://www.mymathlib.com/diffeq/runge-kutta/runge_kutta_3_8.html
    t = t0
    x = copy(x0)
    h = Δt/nSteps
    K1 = similar(x0)
    K2 = similar(x0)
    K3 = similar(x0)
    K4 = similar(x0)
    Ktmp = similar(x0)
    @inbounds for _ in 1:nSteps
        K1 .= f(t, x)

        @. Ktmp = x + h / 3 * K1
        K2 .= f(t + h / 3, Ktmp)

        @. Ktmp = x - h / 3 * K1 + h * K2
        K3 .= f(t + 2 / 3 * h, Ktmp)

        @. Ktmp = x + h * (K1 - K2 + K3)
        K4 .= f(t + h, Ktmp)

        t += h
        @. x += h / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
    end

    return x
end

function odeAux!(dx, dΦ, t, x, Φ, f, Jf)
    dx .= f(t, x)
    mul!(dΦ, Jf(t, x), Φ)
    return nothing
end

function odeCore(t0, x0, Φ0, Δt, f, Jf; nSteps=1)
    t = t0
    x = copy(x0)
    Φ = copy(Φ0)
    h = Δt/nSteps
    K1 = similar(x0);
    K2 = similar(x0);
    K3 = similar(x0);
    K4 = similar(x0);
    Ktmp = similar(x0);
    P1 = similar(Φ0);
    P2 = similar(Φ0);
    P3 = similar(Φ0);
    P4 = similar(Φ0);
    Ptmp = similar(Φ0);
    @inbounds for _ in 1:nSteps
        odeAux!(K1, P1, t, x, Φ, f, Jf)

        @. Ktmp = x + h / 3 * K1
        @. Ptmp = Φ + h / 3 * P1
        odeAux!(K2, P2, t + h / 3, Ktmp, Ptmp, f, Jf)

        @. Ktmp = x - h / 3 * K1 + h * K2
        @. Ptmp = Φ - h / 3 * P1 + h * P2
        odeAux!(K3, P3, t + 2 / 3 * h, Ktmp, Ptmp, f, Jf)

        @. Ktmp = x + h * (K1 - K2 + K3)
        @. Ptmp = Φ + h * (P1 - P2 + P3)
        odeAux!(K4, P4, t + h, Ktmp, Ptmp, f, Jf)

        t += h
        @. x += h / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
        @. Φ += h / 8 * (P1 + 3 * P2 + 3 * P3 + P4)
    end

    return x, Φ
end

"""
    generatePosDefMatrix(n)

Generate a random positive definite matrix of size ```n```.
"""
@inline function generatePosDefMatrix(n)
    P = rand(n, n)
    return (P + P')/2 + n*I
end

"""
    getStd(nav)

Compute the square-root of the diagonal of the navigation covariance matrix ``P``.
"""
@inline function getStd(nav)
    σ = zeros(nav.ns)
    P = getCov(nav)
    @inbounds for i in eachindex(σ)
        σ[i] = sqrt(P[i, i])
    end
    return σ
end

#=
This function decorrelates the measurement noise using the UD factorization.
=#
@inline function decorrelateMeas(y, ŷ, R, H)
    Rc, Rd = UD(R)
    return Rc\y, Rc\ŷ, diagm(Rd), Rc\H
end

"""
    Q = computeQd(Fx, Fw, W, Δt)

Generate the equivalent discrete-time process noise covariance matrix for a
continuous time linear system ``ẋ = F_x x + F_w w``, where ``w`` is a white noise of
power spectral density equal to ``W``.
"""
function computeQd(Fx, Fw, W, Δt)
    Q = Fw * W * Fw'
    n = size(Fx, 1)

    # Exact method for LTI, from: C. Van Loan, Computing integrals
    # involving the matrix exponential, IEEE Transactions on Automatic
    # Control. 23 (3): 395–404, 1978
    G = exp([-Fx Q; zeros(n, n) Fx'] .* Δt)
    Φ = transpose(G[(n + 1):(2 * n), (n + 1):(2 * n)])
    Qd = Φ*G[1:n, (n + 1):(2 * n)]

    return Symmetric(Qd) #(Qd + transpose(Qd))/2
end
