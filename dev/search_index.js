var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = KalmanFilterEngine","category":"page"},{"location":"#KalmanFilterEngine.jl","page":"Home","title":"KalmanFilterEngine.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a set of Kalman filters that can estimate the state of a continuous-time dynamical system given as imput a sequence of discrete-time measurements","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginarrayc\ndot x = f(tx) + w\ny_k = h(t_kx_k) + r_k\nendarray","category":"page"},{"location":"","page":"Home","title":"Home","text":"where w is a zero-mean continuous time white process noise with two-sided power spectral density equal to W,  and r_k is a zero-mean discrete time white measurement noise with covariance equal to R_k. ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Just type:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(url=\"https://github.com/FraCpl/KalmanFilterEngine.jl\")","category":"page"},{"location":"#Capabilities-and-basic-usage","page":"Home","title":"Capabilities and basic usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Four different filters are implemented in KalmanFilterEngine.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"Extended Kalman Filter (EKF)\nUD-factorized Extended Kalman Filter (UDEKF)\nUnscented Kalman Filter (UKF)\nSquare-root Unscented Kalman Filter (SRUKF)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The type of filter can be chosen when initializing the navigation state via the four different initialization functions available: NavState (EKF), NavStateUD (UDEKF), NavStateUKF (UKF), and  NavStateSRUKF (SRUKF).  KalmanFilterEngine.jl provides the two necessary key functions that need to be used to build a Kalman filter: kalmanPropagate! and kalmanUpdate!. Whichever the filter chosen by the user, these two functions will have the same exact interface, to make it easy to swap and test different filters formulations with minimal to no modification to the Kalman filter design.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For example, the following simple single-step Kalman filter","category":"page"},{"location":"","page":"Home","title":"Home","text":"function myKalmanStep!(nav, ty, y)\n    kalmanUpdate!(nav, ty, y, h)\n    kalmanPropagate!(nav, Δt, f, Q)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"can be reused whichever the filter formulation","category":"page"},{"location":"","page":"Home","title":"Home","text":"ty, y = myMeasFun(...)              # Get measurement\n\nnavEKF = NavState(t₀, x₀, P₀)       # Initialize EKF state\nmyKalmanStep!(navEKF, ty, y)        # execute EKF step\n\nnavUKF = NavStateUKF(t₀, x₀, P₀)    # Initialize UKF state\nmyKalmanStep!(navUKF, ty, y)        # Execute UKF step","category":"page"},{"location":"","page":"Home","title":"Home","text":"The key parameters that are needed to define the navigation problem, and that need to be feed to the Kalman filter functions are the system dynamics function, which returns the navigation state vector time derivative:","category":"page"},{"location":"","page":"Home","title":"Home","text":"function f(t, x)\n    ...\n    return ẋ\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"and the measurement function, which returns the predicted measurement y, the measurement noise  covariance R, and the measurement jacobian with respect to the navigation state H (only needed for EKF and UDEKF):","category":"page"},{"location":"","page":"Home","title":"Home","text":"function h(t, x)\n    ...\n    return y, R, H\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"An alternative and convenient way to compute H is to use automatic differentiation capabilities provided by, e.g., ForwardDiff.jl. In this case the user can split the measurement function as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ForwardDiff\n\nfunction myMeas(t, x)\n    ...\n    return y\nend\n\nH(t, x) = ForwardDiff.jacobian(x -> myMeas(t, x), x)\n\nfunction h(t, x)\n    ...\n    return myMeas(t, x), R, H(t, x)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"The user shall also supply the discrete-time process noise covariance matrix Q. This can be easily computed using the computeQd function.","category":"page"},{"location":"","page":"Home","title":"Home","text":"When using EKF or UDEKF, the user has also the option to provide as input the Jacobian of the dynamics Jf(t,x) to the kalmanPropagate! function. If not provided, the function automatically computes the Jacobian using ForwardDiff.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nOne may argue that there is no need to use an UDEKF or a SRUKF when working with Float64. Indeed, \"most smokers do not get cancer or heart disease\".","category":"page"},{"location":"#Working-example","page":"Home","title":"Working example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following code proposes a simple working example to get started with KalmanFilterEngine.jl.  It is based on a double integrator dynamics where 3D position measurements are provided to an Extended Kalman Filter. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"using KalmanFilterEngine, LinearAlgebra, Distributions, Plots\n\n# True state parameters & state transition matrix\nx₀ = zeros(6)                                   # True initial state\nΔt = 0.1                                        # Measurement time step\nΦ = I + [zeros(3,3) Δt*I; zeros(3,6)]           # State transition matrix\n\n# Define navigation problem\nQ = Diagonal([1e-4*ones(3); 1e-3*ones(3)].^2)   # Process noise covariance\nR = 0.0483*Matrix(I,3,3)                        # Measurement noise covariance\nf(t, x) = [x[4:6]; zeros(3)]                    # System dynamics\nh(t, x) = (x[1:3], R, [I zeros(3,3)])           # Measurement equation\n\n# Initialize navigation state\nP₀ = generatePosDefMatrix(6)            # Initial state uncertainty covariance\nx̂₀ = x₀ + rand(MvNormal(P₀))            # Initial estimated state\nnav = NavState(0.0, x̂₀, P₀)\n\n# Define Kalman filter algorithm\nfunction myKalman!(nav, y)\n    kalmanUpdate!(nav, 0.0, y, h)\n    kalmanPropagate!(nav, Δt, f, Q)\nend\n\n# Simulate Kalman filter\nT = []; X = []; X̂ = []; σ = []\nx = x₀\nfor k in 1:100\n    # Generate measurement\n    y = x[1:3] + rand(MvNormal(R))\n\n    # Execute Kalman filter step\n    myKalman!(nav, y)\n\n    # Simulate system dynamics\n    x = Φ*x + rand(MvNormal(Q))\n\n    # Save for post-processing\n    push!(T, nav.t)\n    push!(X, x)\n    push!(X̂, nav.x̂)\n    push!(σ, getStd(nav))\nend\n\n# Plot results for 1st coordinate\nplot(T, getindex.(X,1) - getindex.(X̂,1), lab=\"Nav error\", xlabel=\"Time [s]\")\nplot!(T, +3.0*getindex.(σ,1); color=:red, lab=\"3σ\")\nplot!(T, -3.0*getindex.(σ,1); color=:red, lab=\"\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"The example should produce the following kind of plot: ","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: kff)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Notice how the NavState structure provides a direct access to the estimated state nav.x̂ and the corresponding time nav.t, while the getStd(nav) function returns the square-root of the  diagonal of the covariance matrix of the filter (the full covariance matrix can be retrieved with the  getCov(nav) function).","category":"page"},{"location":"","page":"Home","title":"Home","text":"A more advanced example for a nonlinear orbit estimation problem can be found in the 'examples' folder.","category":"page"},{"location":"#Consider-states","page":"Home","title":"Consider states","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"All filters support consider states, with the assumption that the navigation state defined by the user is already partitioned into solve-for states hat x_s and consider states hat x_c, and that the consider states are  in the tail of the navigation state, i.e., hat x = hat x_s hat x_c. By default, all states are considered to be solve-for states. The number of solve-for states can be modified using the .ns field of the navigation  state:","category":"page"},{"location":"","page":"Home","title":"Home","text":"nav = NavState(t₀, x₀, P₀) \nnav.ns = 6","category":"page"},{"location":"#Error-state-formulation","page":"Home","title":"Error state formulation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"EKF and UDEKF also provide an error state formulation, where the kalmanUpdate! routine can be replaced by  kalmanUpdateError!. This is particularly useful when multiple measurements need to be ingested by the filter during the same update cycle to make the filter insensitive to the order by which measurements are processed.  In this case, the Kalman update procedure needs to be followed by a full state update (generally consisting in adding the error state to the full navigation state) and an error state reset:","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Kalman filter update step\nkalmanUpdateError!(nav, ty1, y1, h1)\nkalmanUpdateError!(nav, ty2, y2, h2)\nkalmanUpdateError!(nav, ty3, y3, h3)\n...\n\n# Update full state and reset error state\nnav.x̂ += nav.δx\nresetErrorState!(nav)\n\n# Kalman filter propagation\n...","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [KalmanFilterEngine]\nOrder   = [:function, :type]\nPages   = [\"ekf.jl\", \"udekf.jl\", \"ukf.jl\", \"srukf.jl\", \"utils.jl\"]","category":"page"},{"location":"#KalmanFilterEngine.getCov-Tuple{NavState}","page":"Home","title":"KalmanFilterEngine.getCov","text":"getCov(nav)\n\nGet navigation covariance matrix P.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.kalmanPropagate!-Tuple{NavState, Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.kalmanPropagate!","text":"kalmanPropagate!(nav, Δt, f, Q; nSteps = 1)\n\nPropagate navigation state forward in time for Δt time units.\n\nInputs include the dynamics function ẋ = f(t, x), and equivalent discrete-time process noise covariance matrix Q. The optional keyword argument nSteps indicates the number of RK4 steps to be performed when numerically integrating the system's dynamics.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.kalmanPropagate!-Tuple{NavState, Vararg{Any, 4}}","page":"Home","title":"KalmanFilterEngine.kalmanPropagate!","text":"kalmanPropagate!(nav, Δt, f, Jf, Q; nSteps = 1)\n\nPropagate navigation state forward in time for Δt time units.\n\nInputs include the dynamics function ẋ = f(t, x), dynamics jacobian function Fx = Jf(t, x), and equivalent discrete-time process noise covariance matrix Q. The optional keyword argument nSteps indicates the number of RK4 steps to be performed when numerically integrating the system's dynamics. This function is only applicable to EKF and UDEKF.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.kalmanUpdate!-Tuple{NavState, Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.kalmanUpdate!","text":"kalmanUpdate!(nav, ty, y, h)\n\nUpdate state of the Kalman filter using the input measurement.\n\nInputs include the measurement time ty, measurement y, measurement equation function ŷ, R, H = h(t, x̂). When using SRUKF or UKF, the measurement function only needs to provide ŷ and R as outputs.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.kalmanUpdateError!-Tuple{NavState, Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.kalmanUpdateError!","text":"kalmanUpdateError!(nav, ty, y, h)\n\nUpdate error state of the Kalman filter using the input measurement.\n\nInputs include the measurement time ty, measurement y, measurement equation function ŷ, R, H = h(t, x̂). This function is only applicable to EKF and UDEKF.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.NavState-Tuple{Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.NavState","text":"NavState(t, x̂, P)\n\nBuild EKF navigation state given as input the initial time, estimated state and navigation covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.NavStateUD-Tuple{Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.NavStateUD","text":"NavStateUD(t, x̂, P)\n\nBuild UDEKF navigation state given as input the initial time, estimated state and navigation covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.NavStateSRUKF-Tuple{Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.NavStateSRUKF","text":"NavStateSRUKF(t, x̂, P)\n\nBuild SRUKF navigation state given as input the initial time, estimated state and navigation covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.NavStateUKF-Tuple{Any, Any, Any}","page":"Home","title":"KalmanFilterEngine.NavStateUKF","text":"NavStateUKF(t, x̂, P)\n\nBuild UKF navigation state given as input the initial time, estimated state and navigation covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.computeQd-NTuple{4, Any}","page":"Home","title":"KalmanFilterEngine.computeQd","text":"Q = computeQd(Fx, Fw, W, Δt)\n\nGenerate the equivalent discrete-time process noise covariance matrix for a continuous time linear system dot x = F_x x + F_w w, where w is a white noise of power spectral density equal to W.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.generatePosDefMatrix-Tuple{Any}","page":"Home","title":"KalmanFilterEngine.generatePosDefMatrix","text":"generatePosDefMatrix(n)\n\nGenerate a random positive definite matrix of size n.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.getStd-Tuple{Any}","page":"Home","title":"KalmanFilterEngine.getStd","text":"getStd(nav)\n\nCompute the square-root of the diagonal of the navigation covariance matrix P.\n\n\n\n\n\n","category":"method"},{"location":"#KalmanFilterEngine.resetErrorState!-Tuple{Any}","page":"Home","title":"KalmanFilterEngine.resetErrorState!","text":"resetErrorState!(nav)\n\nSet error state to zero after the update of an error-state EKF.\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#About-the-author","page":"Home","title":"About the author","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"KalmanFilterEngine.jl was developed by Francesco Capolupo, GNC System Engineer at the European Space Agency, who has more than 10 yearse experience in the  design, analysis, and development of spacecraft GNC subsystems in industry and at the Agency. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"DISCLAIMER: this package has no ties whatsoever with ESA, or any other institution, corporation,  or company, and it only represents a desperate and personal attempt to quit using MATLAB once and for all.","category":"page"}]
}
