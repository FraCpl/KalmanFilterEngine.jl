# KalmanFilterEngine.jl

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://FraCpl.github.io/KalmanFilterEngine.jl/dev/)
[![Build Status](https://github.com/FraCpl/KalmanFilterEngine.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/FraCpl/KalmanFilterEngine.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)


This package provides a set of Kalman filters that can estimate the state of a continuous-time dynamical system
given as imput a sequence of discrete-time measurements
```math
\begin{array}{c}
\dot x = f(t,x) + w\\
y_k = h(t_k,x_k) + r_k
\end{array}
```
where ``w`` is a zero-mean continuous time white process noise with two-sided power spectral density equal to ``W``, 
and ``r_k`` is a zero-mean discrete time white measurement noise with covariance equal to ``R_k``. 

## Installation
Just type:
```julia
using Pkg
Pkg.add(url="https://github.com/FraCpl/KalmanFilterEngine.jl")
```

## Documentation
The full documentation is available [here](https://FraCpl.github.io/KalmanFilterEngine.jl/dev/).