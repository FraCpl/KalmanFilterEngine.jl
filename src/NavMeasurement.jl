abstract type AbstractNavMeasurement end

struct NavMeasurement{T, U} <: AbstractNavMeasurement
    # Measurement model variables
    y::Vector{T}            # Predicted measurement, as computed with x̂ [to be filled by meas. function]
    R::Matrix{T}            # Measurement covariance matrix [to be filled by meas. function]
    H::U                    # Measurement Jacobian [to be filled by meas. function]
    nReject::Int            # Outlier rejection threshold (nσ)

    # Allocation variables
    δy::Vector{T}           # Innovation
    δz::Vector{T}           # Normalized innovation
    Pxy::Matrix{T}          # KF allocation
    Pyy::Matrix{T}          # Innovation covariance
    K::Matrix{T}            # Kalman gain
    Y::Vector{Vector{T}}    # UKF only
end

struct NavMeasurementScalar{T, U} <: AbstractNavMeasurement
    # Measurement model variables
    y::Vector{T}            # Predicted measurement, as computed with x̂ [to be filled by meas. function]
    R::Matrix{T}            # Measurement covariance matrix [to be filled by meas. function]
    H::U                    # Measurement Jacobian [to be filled by meas. function]
    nReject::Int            # Outlier rejection threshold (nσ)

    # Allocation variables
    δy::Vector{T}           # Innovation
    δz::Vector{T}           # Normalized innovation
    Pxy::Vector{T}          # KF allocation
end

# nx is the number of error states
function NavMeasurement(nx::Int, ny::Int; R=zeros(ny, ny), H=zeros(ny, nx), nReject=6)
    y = zeros(ny)
    δy = zeros(ny)
    δz = zeros(ny)
    Pxy = zeros(nx, ny)
    Pyy = zeros(ny, ny)
    K = zeros(nx, ny)
    nX = 2*nx + 1
    Y = [zeros(ny) for _ in 1:nX]
    return NavMeasurement(y, R, H, nReject, δy, δz, Pxy, Pyy, K, Y)
end

# nx is the number of error states
function NavMeasurementScalar(nx::Int, ny::Int; R=zeros(ny, ny), H=zeros(ny, nx), nReject=6)
    y = zeros(ny)
    δy = zeros(ny)
    δz = zeros(ny)
    Pxy = zeros(nx)
    return NavMeasurementScalar(y, R, H, nReject, δy, δz, Pxy)
end
