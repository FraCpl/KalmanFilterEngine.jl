abstract type AbstractNavState end

"""
    NavState(t, x̂, P)

Build navigation state given as input the initial time, estimated
state and navigation covariance matrix.
"""
function NavState(t, x̂, P; type::Symbol=:EKF, α=1e-3, β=2.0, κ=0.0) :: AbstractNavState
    if type == :SRUKF
        return NavStateSRUKF(t, x̂, P; α=α, β=β, κ=κ)
    elseif type == :UD || type == :UDEKF
        return NavStateUD(t, x̂, P)
    elseif type == :UKF
        return NavStateUKF(t, x̂, P; α=α, β=β, κ=κ)
    end
    return NavStateEKF(t, x̂, P)
end
