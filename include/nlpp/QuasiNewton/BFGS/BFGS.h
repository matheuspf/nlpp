#pragma once

#include "../../Helpers/Helpers.h"

#include "../../Helpers/Optimizer.h"

#include "../../Helpers/FiniteDifference.h"

#include "../../LineSearch/StrongWolfe/StrongWolfe.h"


#define CPPOPT_USING_PARAMS_BFGS(...) CPPOPT_USING_PARAMS(__VA_ARGS__);  \
									  using Params::initialHessian;	



namespace nlpp
{

struct BFGS_Constant;
struct BFGS_Diagonal;


namespace params
{

template <class LineSearch = StrongWolfe, class InitialHessian = BFGS_Diagonal>
struct BFGS : public GradientOptimizer<LineSearch>
{
    using GradientOptimizer<LineSearch>::GradientOptimizer;

    InitialHessian initialHessian;
};


} // namespace params



template <class LineSearch = StrongWolfe, class InitialHessian = BFGS_Diagonal>
struct BFGS : public GradientOptimizer<BFGS<LineSearch, InitialHessian>, params::BFGS<LineSearch, InitialHessian>>
{   
    CPPOPT_USING_PARAMS_BFGS(Params, GradientOptimizer<BFGS<LineSearch, InitialHessian>, params::BFGS<LineSearch, InitialHessian>>);

    using Params::Params;


    template <class Function, typename Float, int Rows, int Cols>
    auto optimize (Function f, Eigen::Matrix<Float, Rows, Cols> x0)
    {
        constexpr int Size = Rows * Cols;

        int rows = x0.rows(), cols = x0.cols(), size = rows * cols;

        Eigen::Matrix<Float, Size, Size> In = Eigen::Matrix<Float, Size, Size>::Identity(size, size);

        auto hess = initialHessian([&](const auto& x){ return f.function(x); },
                                   [&](const auto& x){ return f.gradient(x); }, x0);


        Eigen::Matrix<Float, Rows, Cols> x1, g0(rows, cols), g1(rows, cols), dir, s, y;

        Float f0 = f(x0, g0);


        for(int iter = 0; iter < maxIterations; ++iter)
        {
            dir = -hess * g0;

            double alpha = lineSearch(f, x0, dir);

            x1 = x0 + alpha * dir;

            Float f1 = f(x1, g1);

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


} // namespace nlpp