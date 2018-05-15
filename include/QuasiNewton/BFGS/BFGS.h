#pragma once

#include "../../Helpers/Helpers.h"

#include "../../Helpers/FiniteDifference.h"

#include "../../LineSearch/StrongWolfe/StrongWolfe.h"



namespace cppnlp
{

struct BFGS_Constant;
struct BFGS_Diagonal;


template <class LineSearch = StrongWolfe, class InitialHessian = BFGS_Diagonal>
struct BFGS
{
    BFGS (const LineSearch& lineSearch = LineSearch(), const InitialHessian& initialHessian = InitialHessian()) :
          lineSearch(lineSearch), initialHessian(initialHessian) {}

    
    template <class Function, class Gradient>
    Vec operator () (Function function, Gradient gradient, Vec x0)
    {
        int N = x0.rows();

        Mat In = Mat::Identity(N, N);

        Mat hess = initialHessian(function, gradient, x0);

        Vec g0 = gradient(x0);
        Vec x1, g1, dir, s, y;


        for(int iter = 0; iter < maxIter; ++iter)
        {
            dir = -hess * g0;

            double alpha = lineSearch(function, gradient, x0, dir);


            x1 = x0 + alpha * dir;

            g1 = gradient(x1);

            s = x1 - x0;
            y = g1 - g0;


            if(g1.norm() < gTol)
                break;


            double rho = 1.0 / std::max(y.dot(s), constants::eps);

            hess = (In - rho * s * y.transpose()) * hess * (In - rho * y * s.transpose()) + rho * s * s.transpose();

            x0 = x1;
            g0 = g1;
        }

        return x1;
    }


    template <class Function>
    Vec operator () (Function function, const Vec& x)
    {
        return this->operator()(function, fd::gradient(function), x);
    }


    LineSearch lineSearch;
    InitialHessian initialHessian;

    double gTol = 1e-8;
    int maxIter = 1e3;
};




struct BFGS_Diagonal
{
    BFGS_Diagonal (double h = 1e-4) : h(h) {}

    template <class Function, class Gradient>
    Mat operator () (Function function, Gradient gradient, Vec x)
    {
        double fx = function(x);

        Mat hess = Mat::Constant(x.rows(), x.rows(), 0.0);

        hess.diagonal() = (2*h) / (gradient((x.array() + h).matrix()) - gradient((x.array() - h).matrix())).array();

        return hess;
    }
    

    double h;
};


struct BFGS_Constant
{
    BFGS_Constant (double alpha = 1e-4) : alpha(alpha) {}

    template <class Function, class Gradient>
    Mat operator () (Function function, Gradient gradient, const Vec& x0)
    {
        Vec g0 = gradient(x0);
        Vec x1 = x0 - alpha * g0;
        Vec g1 = gradient(x1);

        Vec s = x1 - x0;
        Vec y = g1 - g0;

        Mat hess = (y.dot(s) / y.dot(y)) * Mat::Identity(x0.rows(), x0.rows());

        return hess;
    }


    double alpha;
};


struct BFGS_Identity
{
    template <class Function, class Gradient>
    Mat operator () (Function, Gradient, const Vec& x)
    {
        return Mat::Identity(x.rows(), x.rows());
    }
};


} // namespace cppnlp