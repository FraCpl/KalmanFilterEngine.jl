abstract type AbstractNavState end

"""
    NavState(t, x, P)

Build navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavState(t, x, P; type::Symbol=:EKF, α=1e-3, β=2.0, κ=0.0, iter=5) :: AbstractNavState
    if type == :SRUKF
        return NavStateSRUKF(t, copy(x), copy(P); α=α, β=β, κ=κ)
    elseif type == :UD || type == :UDEKF
        return NavStateUD(t, copy(x), copy(P))
    elseif type == :UKF
        return NavStateUKF(t, copy(x), copy(P); α=α, β=β, κ=κ)
    elseif type == :IEKF
        return NavStateEKF(t, copy(x), copy(P); iter=iter)
    end
    return NavStateEKF(t, copy(x), copy(P); iter=0)
end

getState(nav::AbstractNavState) = copy(nav.x)
