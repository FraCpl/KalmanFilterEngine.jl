abstract type AbstractNavState end

"""
    NavState(t, x, P; type::Symbol=:EKF, ns=size(P,1), kwargs...)

Construct a navigation filter state from time, state vector, and covariance.

This is a unified front-end that dispatches to specific navigation filter implementations
(`EKF`, `UKF`, `SRUKF`, `UD`, etc.) depending on the selected `type`.

# Arguments
- `t`: Current time (scalar).
- `x`: Initial navigation state (estimated state).
- `P`: State covariance matrix.
- `type::Symbol`: Filter type selector. Supported values:
  - `:EKF`              → Extended Kalman Filter
  - `:ESKF`             → Error-State Extended Kalman Filter
  - `:IEKF`             → Iterated Extended Kalman Filter
  - `:UKF`              → Unscented Kalman Filter
  - `:SRUKF`            → Square-root Unscented Kalman Filter
  - `:UD`, `:UDEKF`     → UD-factorized Extended Kalman Filter
- `ns`: number of solve-for states, assumed to be located at x[1:ns].

# Keyword Arguments
Additional keyword arguments are forwarded to specific filter constructors:
- UKF / SRUKF:
  - `α`: Spread parameter (default `1e-3`)
  - `β`: Prior knowledge parameter (default `2.0`)
  - `κ`: Secondary scaling parameter (default `0.0`)

# Behavior
- The covariance matrix `P` is copied and symmetrized before use.
- The state vector `x` is copied to avoid unintended mutation.

# Returns
A concrete navigation state object corresponding to the selected filter type:
`NavStateEKF`, `NavStateUKF`, `NavStateSRUKF`, or `NavStateUD`.

# Examples
```julia
# EKF (default)
ns = NavState(t, x, P)

# UKF with parameters
ns = NavState(t, x, P; type=:UKF, α=1e-2, β=2.0, κ=0.0)
```
"""
function NavState(t, x, P; type::Symbol=:EKF, ns=size(P, 1), kwargs...)
    return NavState(t, copy(x), Symmetric(copy(P)).data, ns, Val(type); kwargs...)
end

NavState(t, x, P, ns, ::Val{:EKF}; kwargs...) = NavStateEKF(t, x, P, ns)
NavState(t, x, P, ns, ::Val{:ESKF}; kwargs...) = NavStateEKF(t, x, P, ns)
NavState(t, x, P, ns, ::Val{:IEKF}; kwargs...) = NavStateEKF(t, x, P, ns)
NavState(t, x, P, ns, ::Val{:UKF}; α=1e-3, β=2.0, κ=0.0) = NavStateUKF(t, x, P, ns; α=α, β=β, κ=κ)
NavState(t, x, P, ns, ::Val{:SRUKF}; α=1e-3, β=2.0, κ=0.0) = NavStateSRUKF(t, x, P, ns; α=α, β=β, κ=κ)
NavState(t, x, P, ns, ::Val{:UD}; kwargs...) = NavStateUD(t, x, P, ns)
NavState(t, x, P, ns, ::Val{:UDEKF}; kwargs...) = NavStateUD(t, x, P, ns)

@inline function getState(nav::T) where {T<:AbstractNavState}
    return copy(nav.x)
end
