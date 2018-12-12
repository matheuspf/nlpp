#pragma once

#include "../../Helpers/Helpers.h"

#include "../../Helpers/Optimizer.h"

#include "../../Helpers/FiniteDifference.h"

#include "../../LineSearch/StrongWolfe/StrongWolfe.h"


#define CPPOPT_USING_PARAMS_BFGS(...) CPPOPT_USING_PARAMS(__VA_ARGS__);  \
									  using Params::initialHessian;	



namespace nlpp
{

template <typename Float = types::Float>
struct BFGS_Constant;

template <typename Float = types::Float>
struct BFGS_Diagonal;


namespace params
{

template <class InitialHessian = BFGS_Diagonal<>, class LineSearch = StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct BFGS : public GradientOptimizer<LineSearch, Stop, Output>
{
    CPPOPT_USING_PARAMS(Params, GradientOptimizer<LineSearch, Stop, Output>);
    using Params::Params;

    InitialHessian initialHessian;
};


} // namespace params



template <class InitialHessian = BFGS_Diagonal<>, class LineSearch = StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct BFGS : public GradientOptimizer<BFGS<InitialHessian, LineSearch, Stop, Output>, 
                                       params::BFGS<InitialHessian, LineSearch, Stop, Output>>
{   
    CPPOPT_USING_PARAMS_BFGS(Params, GradientOptimizer<BFGS<InitialHessian, LineSearch, Stop, Output>, 
                                       params::BFGS<InitialHessian, LineSearch, Stop, Output>>);
    using Params::Params;


    template <class Function, class V>
    V optimize (Function f, V x0)
    {
        using Float = impl::Scalar<V>;

        int rows = x0.rows(), cols = x0.cols(), size = rows * cols;

        impl::Plain2D<V> In = impl::Plain2D<V>::Identity(size, size);

        auto hess = initialHessian(f, x0);

        V x1, g0(rows, cols), g1(rows, cols), dir, s, y;

        Float f0 = f(x0, g0);
        Float f1;

        stop.init(*this, x0, f0, g0);
        output.init(*this, x0, f0, g0);

        for(int iter = 0; iter < stop.maxIterations; ++iter)
        {
            dir = -hess * g0;

            auto alpha = lineSearch(f, x0, dir);

            x1 = x0 + alpha * dir;

            f1 = f(x1, g1);

            s = x1 - x0;
            y = g1 - g0;

            if(stop(*this, x1, f1, g1))
                break;


            Float rho = 1.0 / std::max(y.dot(s), constants::eps_<Float>);

            hess = (In - rho * s * y.transpose()) * hess * (In - rho * y * s.transpose()) + rho * s * s.transpose();

            x0 = x1;
            g0 = g1;

            output(*this, x1, f1, g1);
        }

        output.finish(*this, x1, f1, g1);

        return x1;
    }
};



template <typename Float>
struct BFGS_Diagonal
{
    BFGS_Diagonal (Float h = 1e-4) : h(h) {}

    template <class Function, class Derived>
    impl::Plain2D<Derived> operator () (Function f, const Eigen::MatrixBase<Derived>& x)
    {
        impl::Plain2D<Derived> hess = impl::Plain2D<Derived>::Constant(x.rows(), x.rows(), 0.0);

        hess.diagonal() = (2*h) / (f.gradient((x.array() + h).matrix()) - f.gradient((x.array() - h).matrix())).array();

        return hess;
    }

    Float h;
};


template <typename Float>
struct BFGS_Constant
{
    BFGS_Constant (Float alpha = 1e-4) : alpha(alpha) {}

    template <class Function, class Derived>
    impl::Plain2D<Derived> operator () (Function f, const Eigen::MatrixBase<Derived>& x0)
    {
        auto g0 = f.gradient(x0);
        auto x1 = x0 - alpha * g0;
        auto g1 = f.gradient(x1);

        auto s = x1 - x0;
        auto y = g1 - g0;

        impl::Plain2D<Derived> hess = (y.dot(s) / y.dot(y)) * impl::Plain2D<Derived>::Identity(x0.rows(), x0.rows());

        return hess;
    }

    Float alpha;
};


struct BFGS_Identity
{
    template <class Function, class Derived>
    impl::Plain2D<Derived> operator () (Function, const Eigen::MatrixBase<Derived>& x)
    {
        return impl::Plain2D<Derived>::Identity(x.rows(), x.rows());
    }
};


} // namespace nlpp