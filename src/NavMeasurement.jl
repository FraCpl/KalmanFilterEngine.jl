abstract type AbstractNavMeasurement end

struct NavMeasurement{T, U} <: AbstractNavMeasurement
    # Measurement model variables
    y::Vector{T}        # Predicted measurement, as computed with x̂ [to be filled by meas. function]
    R::Matrix{T}        # Measurement covariance matrix [to be filled by meas. function]
    H::U                # Measurement Jacobian [to be filled by meas. function]
    nReject::Int        # Outlier rejection threshold (nσ)

    # Allocation variables
    δy::Vector{T}       # Innovation
    δz::Vector{T}       # Normalized innovation
    Pxy::Matrix{T}      # KF allocation
    Pyy::Matrix{T}      # Innovation covariance
    K::Matrix{T}        # Kalman gain
end

struct NavMeasurementScalar{T, U} <: AbstractNavMeasurement
    # Measurement model variables
    y::Vector{T}        # Predicted measurement, as computed with x̂ [to be filled by meas. function]
    R::Matrix{T}        # Measurement covariance matrix [to be filled by meas. function]
    H::U                # Measurement Jacobian [to be filled by meas. function]
    nReject::Int        # Outlier rejection threshold (nσ)

    # Allocation variables
    δy::Vector{T}       # Innovation
    δz::Vector{T}       # Normalized innovation
    Pxy::Vector{T}      # KF allocation
end

function NavMeasurement(nδ::Int, ny::Int; R=zeros(ny, ny), H=zeros(ny, nδ), nReject=6)
    y = zeros(ny)
    δy = zeros(ny)
    δz = zeros(ny)
    Pxy = zeros(nδ, ny)
    Pyy = zeros(ny, ny)
    K = zeros(nδ, ny)
    return NavMeasurement(y, R, H, nReject, δy, δz, Pxy, Pyy, K)
end

function NavMeasurementScalar(nδ::Int, ny::Int; R=zeros(ny, ny), H=zeros(ny, nδ), nReject=6)
    y = zeros(ny)
    δy = zeros(ny)
    δz = zeros(ny)
    Pxy = zeros(nδ)
    return NavMeasurementScalar(y, R, H, nReject, δy, δz, Pxy)
end
