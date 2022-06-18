# nlpp

[![Build Status](https://travis-ci.org/matheuspf/nlpp.svg?branch=master)](https://travis-ci.org/matheuspf/nlpp) [![codecov](https://codecov.io/gh/matheuspf/nlpp/branch/master/graph/badge.svg)](https://codecov.io/gh/matheuspf/nlpp)

Include only C++14 nonlinear optimization library






```cpp

// Overloads are resolved based on input types, return types and optimization methods `Conditions` enum

// Function arguments

// Regular ones
opt(Func, ...)
opt(FuncGrad, ...)
opt(Func, Grad, ...)

// In case there is some overlap between func and grad calculation, 
// but they are still more efficient if calculated alone
opt(Func, FuncGrad, ...)
opt(Func, Grad, FuncGrad, ...)
                             
// With hessian
opt(Func, Hess, ...)
opt(FuncGrad, Hess, ...)
opt(Func, Grad, Hess, ...)
opt(Func, FuncGrad, Hess, ...)
opt(Func, Grad, FuncGrad, Hess, ...)



// Domain arguments
//


// Single starting point or no starting point
opt(..., x0, ...)

// Lower/upper bounds
opt(..., lb, ub, ...)
opt(..., x0, lb, ub, ...)

// Linear inequalities
opt(..., A, b, ...)
opt(..., x0, A, b, ...)
opt(..., lb, ub, A, b, ...)
opt(..., x0, lb, ub, A, b, ...)

// Linear equalities
opt(..., Ae, be, ...)
opt(..., x0, Ae, be, ...)
opt(..., lb, ub, Ae, be, ...)
opt(..., x0, lb, ub, Ae, be, ...)

// Linear inequalities + equalities
opt(..., A, b, Ae, be, ...)
opt(..., x0, A, b, Ae, be, ...)
opt(..., lb, ub, A, b, Ae, be, ...)
opt(..., x0, lb, ub, A, b, Ae, be, ...)


// Nonlinear inequalities
model(..., Ineq)
model(..., IneqJac)
model(..., Ineq, Jac)

// Nonlinear equalities
model(..., Eq)
model(..., EqJac)
model(..., Eq, Jac)

// Nonlinear inequalities + equalities (need to figure out this)
model(..., Ineq, Eq)
model(..., IneqJac, EqJac)
model(..., Ineq, Jac, Eq, Jac)
```