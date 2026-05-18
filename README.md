# KalmanFilterEngine.jl

[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://FraCpl.github.io/KalmanFilterEngine.jl/dev/)

A high-performance, allocation-conscious Kalman filtering engine in Julia, designed for real-time and navigation-grade estimation problems.

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

---

# Overview

KalmanFilterEngine.jl provides a flexible framework for implementing:

* Extended Kalman Filters (EKF)
* Error-state Extended Kalman Filters (ESKF)
* Iterated Extended Kalman Filters (IEKF)
* Unscented Kalman Filters (UKF)

The package is designed with a strong emphasis on:

* In-place computation (`!` functions)
* Minimal memory allocations
* Explicit control over mathematical operations
* Separation between state, measurement, and models

---

# Core Concepts

## State Representation

The filter state is stored in a structure such as:

```
NavStateEKF
```

Typical fields include:

* `x` — estimated state vector
* `P` — covariance matrix
* `t` — current time

The design supports both full-state and error-state formulations.

---

## Measurement Model

Measurements are represented using:

```
NavMeasurement
```

Fields:

* `y` — predicted measurement
* `R` — measurement covariance
* `H` — measurement Jacobian (not needed for UKF)
* `δy` — innovation buffer (preallocated)
* `δz` — normalized innovation buffer (preallocated)

Users must provide a measurement function:

```
h!(meas, x, p, t)
```

This function should:

1. Fill the predicted measurement `y`
2. Fill Jacobian `H`
3. Set covariance `R`

All operations should be done in-place.


---

# Design Principles

## In-place Operations

All major functions follow:

```
function foo!(state, ...)
```

This avoids allocations and ensures predictable performance.

---

## Preallocation

Temporary buffers should be reused:

* innovation vectors
* intermediate matrices
* sigma points (if applicable)

Avoid creating arrays inside loops.

---

## Explicit Loops

Small matrix operations are implemented using loops instead of BLAS when beneficial.

Advantages:

* lower overhead
* better cache locality
* easier fusion of operations

---

# Example Usage

```
nav = NavStateEKF(...)
meas = NavMeasurement(...)

# Prediction step (user-defined)
f!(nav, p, t, dt)

# Measurement update
kalmanUpdateIter!(nav, y, h!, meas, p, t; iter=3)
```


# Future Work

Planned or recommended extensions:

* Square-root UKF (SUKF)
* UD Extended Kalman Filter (UDEKF)

---

# Philosophy

KalmanFilterEngine.jl prioritizes:

* Performance over abstraction
* Explicitness over convenience
* Control over automation

It is intended for users who need:

* real-time estimation
* full control over numerical behavior
* integration into larger systems (e.g., navigation stacks)
