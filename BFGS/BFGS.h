#ifndef OPT_BFGS_H
#define OPT_BFGS_H

#include "../../Modelo.h"
#include "../FiniteDifference.h"
#include "../LineSearch/StrongWolfe/StrongWolfe.h"



struct BFGS_Constant;
struct BFGS_Diagonal;


template <class LineSearch = StrongWolfe, class InitialHessian = BFGS_Diagonal>
struct BFGS
{
    //template <class T = LineSearch, class U = InitialHessian, 
    //          enable_if_t<is_same_v<T, StrongWolfe> && is_same_v<U, BFGS_Constant>, int> = 0>
    BFGS (LineSearch lineSearch = LineSearch(), InitialHessian initialHessian = InitialHessian()) :
          lineSearch(lineSearch), initialHessian(initialHessian) {}

    
    template <class Function, class Gradient>
    Vec operator () (Function function, Gradient gradient, Vec x0)
    {
        int N = x0.rows();

        Mat In = Mat::Identity(N, N);

        Mat hess = initialHessian(function, gradient, x0);
        //Mat hess = hessianFD(function)(x0);

        Vec g0 = gradient(x0);
        Vec x1, g1, dir, s, y;

        // db(g0, "\n\n");
        // db(hess); exit(0);


        for(int iter = 0; iter < maxIter; ++iter)
        {
            dir = -hess * g0;

            double alpha = lineSearch(function, gradient, x0, dir);

            // db(iter, "      ", alpha, "      ", function(x0), '\n');
            // db(x0.transpose(), "      ", dir.transpose(), "\n\n\n");
            // FOR(i, N) if(isnan(x0(i))) exit(0);


            x1 = x0 + alpha * dir;

            g1 = gradient(x1);

            s = x1 - x0;
            y = g1 - g0;


            // if(s.dot(y) <= 0.0 || g1.norm() < gTol)
            //    break;

            if(g1.norm() < gTol)
                break;


            double rho = 1.0 / y.dot(s);

            hess = (In - rho * s * y.transpose()) * hess * (In - rho * y * s.transpose()) + rho * s * s.transpose();

            x0 = x1;
            g0 = g1;
        }

        return x1;
    }


    template <class Function>
    Vec operator () (Function function, const Vec& x)
    {
        return this->operator()(function, gradientFD(function), x);
    }


    LineSearch lineSearch;
    InitialHessian initialHessian;

    double gTol = 1e-8;
    int maxIter = 1e3;
};




struct BFGS_Diagonal
{
    BFGS_Diagonal (double h0 = 1e-8) : h0(h0) {}

    template <class Function, class Gradient>
    Mat operator () (Function function, Gradient gradient, Vec x)
    {
        double fx = function(x), h = h0;

        if(fx < 1e-6 || fx > 1e-6)
            h = 1e-4;

        Mat hess = Mat::Constant(x.rows(), x.rows(), 0.0);

        hess.diagonal() = (2*h) / (gradient((x.array() + h).matrix()) - gradient((x.array() - h).matrix())).array();

        //DB(hess.diagonal()); exit(0);

        return hess;
    }

    

    double h0;
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

        DB(hess);

        return hess;
    }


    double alpha;
};



#endif // OPT_BFGS_H