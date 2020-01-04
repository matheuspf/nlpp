#pragma once

#include "../../helpers/helpers.hpp"

#include "../../Helpers/Optimizer.h"

#include "../../Helpers/FiniteDifference.h"

#include "../../LineSearch/StrongWolfe/StrongWolfe.hpp"


#define CPPOPT_USING_PARAMS_BFGS(...) CPPOPT_USING_PARAMS(__VA_ARGS__);  \
									  using Params::initialHessian;	



namespace nlpp
{

template <typename Float = types::Float>
struct BFGS_Constant;

template <typename Float = types::Float>
struct BFGS_Diagonal;


namespace impl
{

namespace params
{

template <class Params_, class InitialHessian = BFGS_Diagonal<>>
struct BFGS : public Params_
{
    CPPOPT_USING_PARAMS(Params, Params_);
    using Params::Params;

    InitialHessian initialHessian;
};

} // namespace params



template <class Params_, class InitialHessian = BFGS_Constant<>>
struct BFGS : public params::BFGS<Params_, InitialHessian>
{
    CPPOPT_USING_PARAMS_BFGS(Params, params::BFGS<Params_, InitialHessian>);
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

        for(int iter = 0; iter < stop.maxIterations(); ++iter)
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

        return x1;
    }
};

} // namespace impl

template <class InitialHessian = BFGS_Diagonal<>, class LineSearch = StrongWolfe<>,
          class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<0>>
struct BFGS : public impl::BFGS<params::LineSearchOptimizer<LineSearch, Stop, Output>, InitialHessian>,
			  public GradientOptimizer<BFGS<InitialHessian, LineSearch, Stop, Output>>
{
    CPPOPT_USING_PARAMS(Impl, impl::BFGS<params::LineSearchOptimizer<LineSearch, Stop, Output>, InitialHessian>);
    using Impl::Impl;

    template <class Function, class V>
    V optimize (Function f, V x)
    {
        return Impl::optimize(f, x);
    }
};


namespace poly
{

template <class InitialHessian = BFGS_Constant<>, class V = ::nlpp::Vec>
struct BFGS : public ::nlpp::impl::BFGS<::nlpp::poly::GradientOptimizer<V>, InitialHessian>
{
	CPPOPT_USING_PARAMS(Impl, ::nlpp::impl::BFGS<::nlpp::poly::GradientOptimizer<V>, InitialHessian>);
	using Impl::Impl;

	virtual V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
	{
		return Impl::optimize(f, x);
	}

	virtual BFGS* clone_impl () const { return new BFGS(*this); }
};

} // namespace poly


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



namespace out
{

template <typename Float = types::Float>
struct BFGS : public GradientOptimizer<1, Float>
{
    using Base = GradientOptimizer<1, Float>;
    using Base::Base;
    using Base::init;
    using Base::operator();
    using Base::finish;

    template <class Params, class InitialHessian, class V>
    void operator() (const ::nlpp::impl::params::BFGS<Params, InitialHessian>& optimizer,
                     const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx)
    {
        Base::operator()(optimizer, x, fx, gx);
        std::cout << optimizer.hess << "\n\n\n";
    }
};


namespace poly
{

template <class V = ::nlpp::Vec>
struct BFGS : public GradientOptimizerBase<V>,
              public ::nlpp::out::BFGS<::nlpp::impl::Scalar<V>>
{
    using Float = ::nlpp::impl::Scalar<V>;
    using Impl = ::nlpp::out::BFGS<::nlpp::impl::Scalar<V>>;

    virtual void init (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        Impl::init(optimizer, x, fx, gx);
    }

    virtual void operator() (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return Impl::operator()(static_cast<const ::nlpp::impl::params::BFGS<::nlpp::poly::GradientOptimizer<V>>&>(optimizer), x, fx, gx);
    }

    virtual void finish (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        Impl::finish(optimizer, x, fx, gx);
    }

    virtual BFGS* clone_impl () const { return new BFGS(*this); }
};


} // namespace poly

} // namespace out




} // namespace nlpp