using KalmanFilterEngine
using BenchmarkTools

function main()
    t0 = 0
    x0 = randn(12)
    Δt = 1.0
    f!(dx, x, p, t) = @inbounds for i in eachindex(dx); dx[i] = randn(); end
    Jf!(F, x, p, t) = @inbounds for i in eachindex(F); F[i] = randn(); end

    oc = KalmanFilterEngine.ODECache(x0)
    p = nothing

    @btime KalmanFilterEngine.odeCore!($x0, $t0, $Δt, $f!, $Jf!, $p, $oc)
    return nothing
end
main()
