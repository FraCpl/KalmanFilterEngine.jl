abstract type AbstractNavState end

"""
    NavState(t, x, P, ns)

Build navigation state given as input the initial time, estimated
state and navigation covariance matrix. The optional parameter 'ns'
indicates the number of solve-for components of x, assuming x[1:ns]
as indices of the solve-for states.
"""
function NavState(t, x, P, ns=size(P, 1); type::Symbol=:EKF, α=1e-3, β=2.0, κ=0.0)
    Psym = 0.5 * (P + transpose(P))
    if type == :SRUKF
        return NavStateSRUKF(t, copy(x), Psym, ns; α=α, β=β, κ=κ)
    elseif type == :UD || type == :UDEKF
        return NavStateUD(t, copy(x), Psym, ns)
    elseif type == :UKF
        return NavStateUKF(t, copy(x), Psym, ns; α=α, β=β, κ=κ)
    end
    return NavStateEKF(t, copy(x), Psym, ns)
end

@inline function getState(nav::T) where {T<:AbstractNavState}
    return copy(nav.x)
end
