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
