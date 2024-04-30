module KalmanFilterEngine

using LinearAlgebra, ForwardDiff

export generatePosDefMatrix, resetErrorState!, getStd, computeQd
include("utils.jl")

export NavState
export getCov, kalmanUpdate!, kalmanPropagate!, kalmanUpdateError!
include("NavState.jl")
include("ekf.jl")
include("udekf.jl")
include("ukf.jl")
include("srukf.jl")

end
