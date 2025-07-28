using KalmanFilterEngine
using LinearAlgebra
using BenchmarkTools

nav = NavState(0.0, randn(8), generatePosDefMatrix(8))
nav.ns = 6

nav2 = deepcopy(nav)

y = [0.314; 12.00234; -3.3023]
yest = [0.214; 7.1234; -2.343]
R = 1e-3*I(3)
H = [I zeros(3, 5)]
δy = zero(y)
δz = zero(y)
Pxy = Vector{eltype(nav.P)}(undef, nav.nδ)                         # Save allocations
Ks = Vector{eltype(nav.P)}(undef, nav.ns)                          # Save allocations
KPyyK = Matrix{eltype(nav.P)}(undef, nav.ns, nav.ns)               # Save allocations
KPxyT = Matrix{eltype(nav.P)}(undef, nav.ns, nav.nδ - nav.ns)      # Save allocations

@btime kalmanUpdateErrorScalar!($nav, $y, $yest, $R, $H, $δy, $δz, $Pxy, $Ks, $KPyyK, $KPxyT)

Pxy = Matrix{eltype(nav.P)}(undef, nav.nδ, length(y))              # Save allocations
Pyy = Matrix{eltype(nav.P)}(undef, size(R))                        # Save allocations
xs = Vector{Float64}(undef, nav.ns)                                 # Save allocations
PyyK = Matrix{eltype(nav.P)}(undef, length(y), nav.ns)             # Save allocations
KPyyK = Matrix{eltype(nav.P)}(undef, nav.ns, nav.ns)               # Save allocations
KPxyT = Matrix{eltype(nav.P)}(undef, nav.ns, nav.nδ - nav.ns)      # Save allocations

@btime kalmanUpdateError!($nav, $y, $yest, $R, $H, $δy, $δz, $Pxy, $Pyy, $xs, $PyyK, $KPyyK, $KPxyT)
