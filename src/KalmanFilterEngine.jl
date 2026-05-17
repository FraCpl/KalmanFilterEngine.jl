# Author: F. Capolupo
# European Space Agency, 2024
module KalmanFilterEngine

using LinearAlgebra

export generatePosDefMatrix, getStd, computeQd
include("ode.jl")
include("utils.jl")

export NavState, AbstractNavState, getState
export getCov, kalmanPropagate!, kalmanPropagateCov!, transformCov!
export kalmanUpdate!, kalmanUpdateError!, kalmanUpdateIter!, kalmanUpdateScalar!, kalmanUpdateErrorScalar!
export NavMeasurement, NavMeasurementScalar
include("NavMeasurement.jl")
include("NavState.jl")
include("ekf.jl")
include("iekf.jl")
include("udekf.jl")
include("ukf.jl")
include("srukf.jl")

end
