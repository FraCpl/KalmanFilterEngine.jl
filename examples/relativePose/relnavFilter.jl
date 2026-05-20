struct NavData
    μ::Float64
    n::Float64
    rT::Float64
    Δt::Float64
    qTmp::Vector{Float64}
    mJ1::Matrix{Float64}
    mJ2::Matrix{Float64}
    mJ3::Matrix{Float64}
    JT_T::Matrix{Float64}
    invJT_T::Matrix{Float64}
    J::Matrix{Float64}
    Fw::Matrix{Float64}
    Q::Matrix{Float64}
    R_SC::Matrix{Float64}
    posCS_S::Vector{Float64}
    Mlos::Matrix{Float64}
    WIT_T::Matrix{Float64}
    tmp1::Vector{Float64}
    tmp2::Vector{Float64}
    tmp3::Vector{Float64}
    tmp4::Matrix{Float64}
    tmp5::Matrix{Float64}
    tmp6::Matrix{Float64}
    tmp7::Matrix{Float64}
    meas::NavMeasurementScalar
end

function NavData(μ, n, Δt, wv, wω, stdPos, stdLos, R_SC, posCS_C)
    rT = (μ/n^2)^(1/3)
    qTmp = zeros(4)
    mJ1 = zeros(3, 6)
    mJ2 = zeros(3, 6)
    mJ3 = zeros(3, 6)
    JT_T = zeros(3, 3)
    invJT_T = zeros(3, 3)
    J = zeros(21, 21)
    Fw = [zeros(3, 6); wv*I(3) zeros(3, 3); zeros(3, 6); zeros(3, 3) wω*I(3); zeros(9, 6)]
    Mlos = zeros(2, 3)
    Q = zeros(21, 21)
    posCS_S = R_SC * posCS_C

    meas = NavMeasurementScalar(21, 2; R=stdLos^2*Matrix(I(2)))

    nd = NavData(μ, n, rT, Δt, qTmp, mJ1, mJ2, mJ3, JT_T, invJT_T, J, Fw, Q, R_SC, posCS_S,
        Mlos, zeros(3, 3), zeros(3), zeros(3), zeros(3), zeros(3, 3), zeros(3, 3), zeros(3, 3), zeros(3, 6), meas)

    nd.Q .= navProcessNoise(nd)
    return nd
end

function multJ!(mJ, X)
    mJ .= 0.0
    mJ[1, 1] = X[1]; mJ[1, 2] = X[2]; mJ[1, 3] = X[3]
    mJ[2, 2] = X[1]; mJ[2, 4] = X[2]; mJ[2, 5] = X[3]
    mJ[3, 3] = X[1]; mJ[3, 5] = X[2]; mJ[3, 6] = X[3]
    return mJ
end

function invJ!(iJ, J)
    jxx, jxy, jxz, _, jyy, jyz, _, _, jzz = J
    detj = 1 / (jxx*(jyy*jzz - (jyz^2)) - jxy*(jxy*jzz - jxz*jyz) + (jxy*jyz - jxz*jyy)*jxz)
    iJ[1, 1] = (jyy*jzz - (jyz^2)) * detj
    iJ[1, 2] = (-jxy*jzz + jxz*jyz) * detj
    iJ[1, 3] = (jxy*jyz - jxz*jyy) * detj
    iJ[2, 1] = iJ[1, 2]
    iJ[2, 2] = (jxx*jzz - (jxz^2)) * detj
    iJ[2, 3] = (-jxx*jyz + jxy*jxz) * detj
    iJ[3, 1] = iJ[1, 3]
    iJ[3, 2] = iJ[2, 3]
    iJ[3, 3] = (jxx*jyy - (jxy^2)) * detj
    return iJ
end

function updateQuat(q, δθ, qTmp)
    c = 1 / sqrt(4 + dot(δθ, δθ))
    qTmp[1] = 2 * c
    @inbounds for i in 2:4
        qTmp[i] = c * δθ[i - 1]
    end
    return q_multiply(q, qTmp)
end

function updateNavState!(navData::NavData, x, δx)
    if !all(iszero, δx)
        qTmp = navData.qTmp
        for ix in 1:6
            x[ix] += δx[ix]
        end
        x[7:10] .= updateQuat(x[7:10], δx[7:9], qTmp)
        for ix in 11:lastindex(x)
            x[ix] += δx[ix-1]
        end
        δx .= 0.0
    end
    return x
end

# xEst = [posTC_L; velTC_L; q_IT; ωIT_T;  JT_T; posTQ_Q]
# error =[    1:3;     4:6;  7:9; 10:12; 13:18;  19:21]
# full = [    1:3;     4:6; 7:10; 11:13; 14:19;  20:22]
function navDyn!(dX, X, navData::NavData, t)
    μ = navData.μ
    n = navData.n
    JT_T = navData.JT_T
    rT = navData.rT

    # Extract states from state vector
    x, y, z = X[1], X[2], X[3]    # posTC_L
    vx, vy, vz = X[4], X[5], X[6]        # velTC_L
    q_IT = X[7:10]
    ωIT_T = X[11:13]
    JT_T[1, 1] = X[14]
    JT_T[1, 2] = X[15]
    JT_T[1, 3] = X[16]
    JT_T[2, 2] = X[17]
    JT_T[2, 3] = X[18]
    JT_T[3, 3] = X[19]
    JT_T[2, 1] = JT_T[1, 2]
    JT_T[3, 1] = JT_T[1, 3]
    JT_T[3, 2] = JT_T[2, 3]

    # Translational non-linear relative dynamics
    zC = z - rT
    rC = sqrt(x * x + y * y + zC * zC)
    irC3 = 1 / rC^3

    dX[1] = vx; dX[2] = vy; dX[3] = vz
    dX[4] = 2*n*vz + (n^2)*x - μ * x * irC3
    dX[5] = -μ * y * irC3
    dX[6] = -2*n*vx + (n^2)*z - μ *(1 / rT^2 + zC * irC3)

    # dx[4] = 2*n*vz
    # dx[5] = - n^2 * y
    # dx[6] = -2*n*vx + 3(n^2)*z

    # Target absolute rotational kinematics
    qTmp = navData.qTmp
    q_derivative!(qTmp, q_IT, ωIT_T)
    @inbounds for i in 1:4
        dX[i+6] = qTmp[i]
    end

    # Target absolute rotational dynamics
    tmp1 = navData.tmp1; tmp2 = navData.tmp2
    invJT_T = invJ!(navData.invJT_T, JT_T)
    mul!(tmp1, JT_T, ωIT_T)     # Jω
    cross!(tmp2, tmp1, ωIT_T)   # (Jω × ω)
    mul!(tmp1, invJT_T, tmp2)   # ̇ω = J⁻¹(Jω × ω)
    dX[11] = tmp1[1]
    dX[12] = tmp1[2]
    dX[13] = tmp1[3]

    return dX
end

navDynJacobian(X, navData, t) = navDynJacobian!(navData.J, X, navData, t)

function navDynJacobian!(J, X, navData, t)
    n = navData.n
    # J = navData.J
    WIT_T = navData.WIT_T
    JT_T = navData.JT_T

    J[1, 4] = 1.0; J[2, 5] = 1.0; J[3, 6] = 1.0
    J[4, 6] = 2*n
    J[5, 2] = -n^2
    J[6, 3] = 3*n^2
    J[6, 4] = -2*n

    ωIT_T = X[11:13]
    crossMat!(WIT_T, ωIT_T)
    J[7:9, 7:9] = -WIT_T
    J[7, 10] = 1.0; J[8, 11] = 1.0; J[9, 12] = 1.0

    JT_T[1, 1] = X[14]
    JT_T[1, 2] = X[15]
    JT_T[1, 3] = X[16]
    JT_T[2, 2] = X[17]
    JT_T[2, 3] = X[18]
    JT_T[3, 3] = X[19]
    JT_T[2, 1] = JT_T[1, 2]
    JT_T[3, 1] = JT_T[1, 3]
    JT_T[3, 2] = JT_T[2, 3]

    if !all(iszero, JT_T)
        tmp1 = navData.tmp1; tmp2 = navData.tmp2; αIT_T = navData.tmp3
        tmp5 = navData.tmp5; tmp6 = navData.tmp6; tmp7 = navData.tmp7
        tmp4 = navData.tmp4
        invJT_T = invJ!(navData.invJT_T, JT_T)

        mul!(tmp1, JT_T, ωIT_T)
        crossMat!(tmp4, tmp1)
        cross!(tmp2, tmp1, ωIT_T)  # Jω × ω
        mul!(tmp5, WIT_T, JT_T)
        tmp4 .-= tmp5
        mul!(tmp6, invJT_T, tmp4)
        @inbounds for j in 1:3, i in 1:3
            J[i+9, j+9] = tmp6[i, j]
        end
        # J[10:12, 10:12] = JT_T \ (crossMat(JT_T * ωIT_T) - crossMat(ωIT_T) * JT_T)

        mul!(αIT_T, invJT_T, tmp2)  # ̇ω = J⁻¹(Jω × ω)
        mJ1 = multJ!(navData.mJ1, αIT_T)
        mJ2 = multJ!(navData.mJ2, ωIT_T)
        mJ3 = navData.mJ3
        mul!(mJ3, WIT_T, mJ2)
        mJ1 .+= mJ3
        mul!(tmp7, invJT_T, mJ1)
        @inbounds for j in 1:6, i in 1:3
            J[i+9, j+12] = -tmp7[i, j]
        end
        # J[10:12, 13:18] = - JT_T \ (multJ(αIT_T) + crossMat(ωIT_T) * multJ(ωIT_T))
    end
    return J
end

navProcessNoise(navData, x=zeros(22)) = computeQd(navDynJacobian(x, navData, 0.0), navData.Fw, I, navData.Δt)

function losMeas!(navData::NavData, X, posQF_Q, R_CI, R_IL)
    meas = navData.meas
    posSF_S, Hpos = posMeas(navData, X, posQF_Q, R_CI, R_IL)
    xSF_S, ySF_S, zSF_S = posSF_S
    meas.y[1] = xSF_S / zSF_S
    meas.y[2] = ySF_S / zSF_S

    # Compute jacobian
    Mlos = navData.Mlos
    Mlos[1, 1] = 1/zSF_S
    Mlos[1, 3] = -xSF_S/zSF_S^2
    Mlos[2, 2] = 1/zSF_S
    Mlos[2, 3] = -ySF_S/zSF_S^2
    mul!(meas.H, Mlos, Hpos)
    return meas.y
end

function posMeas(navData::NavData, X, posQF_Q, R_CI, R_IL)
    # Compute measurement
    posTC_L = X[1:3]
    q_IT = X[7:10]
    posTQ_Q = X[20:22]

    R_ST = navData.R_SC * R_CI * q_toDcm(q_IT)
    R_SL = navData.R_SC * R_CI * R_IL

    posTF_S = navData.tmp1
    posTC_S = navData.tmp3
    posTF_Q = posTQ_Q + posQF_Q
    mul!(posTF_S, R_ST, posTF_Q)
    mul!(posTC_S, R_SL, posTC_L)

    posSF_S = [posTF_S[i] - navData.posCS_S[i] - posTC_S[i] for i in 1:3]

    # Compute jacobian
    xTF_Q = navData.tmp4
    H = zeros(3, 21)#navData.Hpos
    crossMat!(xTF_Q, posTF_Q)
    mul!(navData.tmp5, R_ST, xTF_Q)
    @inbounds for j in 1:3, i in 1:3
        H[i, j] = -R_SL[i, j]
        H[i, j+6] = -navData.tmp5[i, j]
        H[i, i+18] = R_ST[i, j]
    end

    return posSF_S, H
end

# Define Kalman filter
function kalmanFilter!(navState, navData, y, R_CI, R_IL)
    # Update step at t[k-1] with y[k-1]
    for yk in y
        losMeas!(navData, navState.x, yk.posQF_Q, R_CI, R_IL)
        kalmanUpdateError!(navState, yk.yMeas, navData.meas)
    end

    # Update full state
    updateNavState!(navData, navState.x, navState.δx)
    navState.P .= 0.5 .* (navState.P + navState.P')

    # Propagate state from t[k-1] to t[k]
    kalmanPropagate!(navState, navData.Δt, navDyn!, navDynJacobian!, navData, navData.Q; nSteps=5)
    return nothing
end
