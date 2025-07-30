using KalmanFilterEngine
using LinearAlgebra
using BenchmarkTools

function testalloc()
    nav = NavState(0.0, randn(8), generatePosDefMatrix(8), 6)

    y = [0.314; 12.00234; -3.3023]
    yest = [0.214; 7.1234; -2.343]
    R = 1e-3*I(3)
    H = [I zeros(3, 5)]
    δy = zero(y)
    δz = zero(y)

    @btime kalmanUpdateErrorScalar!($nav, $y, $yest, $R, $H, $δy, $δz)

    Pxy = Matrix{eltype(nav.P)}(undef, nav.nδ, length(y))              # Save allocations
    Pyy = Matrix{eltype(nav.P)}(undef, size(R))                        # Save allocations
    PyyK = Matrix{eltype(nav.P)}(undef, length(y), nav.ns)             # Save allocations

    @btime kalmanUpdateError!($nav, $y, $yest, $R, $H, $δy, $δz, $Pxy, $Pyy, $PyyK)
    return
end

testalloc()
