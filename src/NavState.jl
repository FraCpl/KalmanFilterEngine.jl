abstract type AbstractNavState end

"""
    NavState(t, x, P, ns)

Build navigation state given as input the initial time, estimated
state and navigation covariance matrix. The optional parameter 'ns'
indicates the number of solve-for components of x, assuming x[1:ns]
as indices of the solve-for states.
"""
function NavState(t, x, P, ns=size(P, 1); type::Symbol=:EKF, α=1e-3, β=2.0, κ=0.0) :: AbstractNavState
    if type == :SRUKF
        return NavStateSRUKF(t, copy(x), copy(P), ns; α=α, β=β, κ=κ)
    elseif type == :UD || type == :UDEKF
        return NavStateUD(t, copy(x), copy(P), ns)
    elseif type == :UKF
        return NavStateUKF(t, copy(x), copy(P), ns; α=α, β=β, κ=κ)
    end
    return NavStateEKF(t, copy(x), copy(P), ns)
end

@inline getState(nav::AbstractNavState) = copy(nav.x)
