# Author: F. Capolupo
# European Space Agency, 2024
module KalmanFilterEngine

using LinearAlgebra#, ForwardDiff

export generatePosDefMatrix, getStd, computeQd
include("ode.jl")
include("utils.jl")

export NavState, AbstractNavState, getState
export getCov, kalmanPropagate!, kalmanPropagateCov!
export kalmanUpdate!, kalmanUpdateError!, kalmanUpdateIter!, kalmanUpdateScalar!, kalmanUpdateErrorScalar!
include("NavState.jl")
include("ekf.jl")
include("udekf.jl")
include("ukf.jl")
include("srukf.jl")

end
