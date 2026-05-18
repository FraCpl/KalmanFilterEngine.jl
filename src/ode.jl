# 3/8 Runge-Kutta Method
# http://www.mymathlib.com/diffeq/runge-kutta/runge_kutta_3_8.html
struct ODECache{T, D}
    K1::T; K2::T; K3::T; K4::T; Ktmp::T
    P1::D; P2::D; P3::D; P4::D; Ptmp::D
    Jtmp::D; Φ::D
end

function ODECache(x, P)
    K1 = zero(x); K2 = zero(x); K3 = zero(x); K4 = zero(x); Ktmp = zero(x)
    P1 = zero(P); P2 = zero(P); P3 = zero(P); P4 = zero(P); Ptmp = zero(P)
    Jtmp = zero(P); Φ = zero(P)
    return ODECache(K1, K2, K3, K4, Ktmp, P1, P2, P3, P4, Ptmp, Jtmp, Φ)
end

ODECache(x) = ODECache(x, x * x')

# f!(dx, x, p, t)
# x(t) -> x(t + Δt)
function odeSolve!(x, t, Δt, f!, p, odeCache::ODECache{T, D}; nSteps=1) where {T, D}
    h = Δt / nSteps
    K1 = odeCache.K1; K2 = odeCache.K2; K3 = odeCache.K3;
    K4 = odeCache.K4; Ktmp = odeCache.Ktmp
    @inbounds for _ in 1:nSteps
        f!(K1, x, p, t)

        @. Ktmp = x + h / 3 * K1
        f!(K2, Ktmp, p, t + h / 3)

        @. Ktmp = x - h / 3 * K1 + h * K2
        f!(K3, Ktmp, p, t + 2 / 3 * h)

        @. Ktmp = x + h * (K1 - K2 + K3)
        f!(K4, Ktmp, p, t + h)

        t += h
        @. x += h / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
    end
    return nothing
end

# f!(dx, x, p, t)       dx/dt = f(t, x)
# Jf!(Fx, x, p, t)      where Fx = ∂f(x) / ∂x
# x(t) -> x(t + Δt)
# Φ = Φ(t + Δt, t)
function odeSolve!(x, t, Δt, f!, Jf!, p, odeCache::ODECache{T, D}; nSteps=1) where {T, D}
    Φ = odeCache.Φ
    fill!(Φ, 0)
    @inbounds for i in axes(Φ, 1)
        Φ[i, i] = 1
    end
    h = Δt / nSteps
    K1 = odeCache.K1; K2 = odeCache.K2; K3 = odeCache.K3;
    K4 = odeCache.K4; Ktmp = odeCache.Ktmp
    P1 = odeCache.P1; P2 = odeCache.P2; P3 = odeCache.P3;
    P4 = odeCache.P4; Ptmp = odeCache.Ptmp; Jtmp = odeCache.Jtmp
    @inbounds for _ in 1:nSteps
        f!(K1, x, p, t)
        Jf!(Jtmp, x, p, t)
        mul!(P1, Jtmp, Φ)

        @. Ktmp = x + h / 3 * K1
        @. Ptmp = Φ + h / 3 * P1
        f!(K2, Ktmp, p, t + h / 3)
        Jf!(Jtmp, Ktmp, p, t + h / 3)
        mul!(P2, Jtmp, Ptmp)

        @. Ktmp = x - h / 3 * K1 + h * K2
        @. Ptmp = Φ - h / 3 * P1 + h * P2
        f!(K3, Ktmp, p, t + 2 / 3 * h)
        Jf!(Jtmp, Ktmp, p, t + 2 / 3 * h)
        mul!(P3, Jtmp, Ptmp)

        @. Ktmp = x + h * (K1 - K2 + K3)
        @. Ptmp = Φ + h * (P1 - P2 + P3)
        f!(K4, Ktmp, p, t + h)
        Jf!(Jtmp, Ktmp, p, t + h)
        mul!(P4, Jtmp, Ptmp)

        t += h
        @. x += h / 8 * (K1 + 3 * K2 + 3 * K3 + K4)
        @. Φ += h / 8 * (P1 + 3 * P2 + 3 * P3 + P4)
    end
    return nothing
end
