function odeCore(t0, x0, Δt, f; nSteps=1)
    # 3/8 Runge-Kutta Method
    # http://www.mymathlib.com/diffeq/runge-kutta/runge_kutta_3_8.html
    t = copy(t0)
    x = copy(x0)
    h = Δt/nSteps
    K1 = similar(x0); K2 = similar(x0); K3 = similar(x0); K4 = similar(x0)
    for _ in 1:nSteps
        K1 .= h*f(t, x)
        K2 .= h*f(t + 1/3*h, x + K1/3)
        K3 .= h*f(t + 2/3*h, x - K1/3 + K2)
        K4 .= h*f(t + h, x + K1 - K2 + K3)
        t += h
        x .+= (K1 + 3K2 + 3K3 + K4)/8
    end

    return x
end

"""
    generatePosDefMatrix(n)

Generate a random positive definite matrix of size ```n```.
"""
function generatePosDefMatrix(n)
    P = rand(n, n)
    return (P + P')/2 + Matrix(n*I, n, n)
end

"""
    resetErrorState!(nav)

Set error state to zero after the update of an error-state EKF.
"""
function resetErrorState!(nav)
    nav.δx .= 0.0
end

"""
    getStd(nav)

Compute the square-root of the diagonal of the navigation covariance matrix ``P``.
"""
function getStd(nav)
    return sqrt.(diag(getCov(nav)))
end

#=
This function decorrelates the measurement noise using the UD factorization.
=#
function decorrelateMeas(y, ŷ, R, H)
    Rc, Rd = UD(R)

    return Rc\y, Rc\ŷ, diagm(Rd), Rc\H
end

"""
    Q = computeQd(Fx, Fw, W, Δt)

Generate the equivalent discrete-time process noise covariance matrix for a
continuous time linear system ``\\dot x = F_x x + F_w w``, where ``w`` is a white noise of
power spectral density equal to ``W``.
"""
function computeQd(Fx, Fw, W, Δt)
    Q = Fw*W*Fw'
    n = size(Fx,1)

    # Exact method for LTI, from: C. Van Loan, Computing integrals
    # involving the matrix exponential, IEEE Transactions on Automatic
    # Control. 23 (3): 395–404, 1978
    G = exp([-Fx Q; zeros(n, n) Fx'].*Δt)
    Φ = transpose(G[n+1:2*n, n+1:2*n])
    Qd = Φ*G[1:n, n+1:2*n]

    return Symmetric(Qd) #(Qd + transpose(Qd))/2
end
