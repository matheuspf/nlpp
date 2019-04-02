#pragma once

#include "Helpers.h"


namespace nlpp
{

namespace stop
{

namespace impl
{

template <class Impl, typename Float>
struct GradientOptimizer
{
    GradientOptimizer(int maxIterations_ = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) : 
                      maxIterations_(maxIterations_), xTol(xTol), fTol(fTol), gTol(gTol), initialized(false) {}

    void initialize ()
    {
        initialized = false;
    }


    template <class Stop, class Output, class V>
    bool operator () (const params::Optimizer<Stop, Output>& optimizer,
                      const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx) 
    {
        bool doStop = false;

        if(initialized)
        {
            bool fStop = std::abs(fx - fx0) < fTol;
            bool xStop = (::nlpp::impl::cast<Float>(x) - x0).norm() < xTol;
            bool gStop = gx.norm() < gTol;

            doStop = static_cast<Impl&>(*this).stop(xStop, fStop, gStop);
        }

        fx0 = fx;
        x0 = ::nlpp::impl::cast<Float>(x);
        gx0 = ::nlpp::impl::cast<Float>(gx);
        initialized = true;

        return doStop;
    }


    int maxIterations () { return maxIterations_; }



    Float fx0;

    VecX<Float> x0;

    VecX<Float> gx0;


    int maxIterations_;      ///< Maximum number of outer iterations

	Float xTol;            ///< Minimum tolerance on the norm of the input (@c x) between iterations

	Float fTol;            ///< Minimum tolerance on the value of the function (@c x) between iterations

    Float gTol;            ///< Minimum tolerance on the norm of the gradient (@c g) between iterations

    bool initialized;
};


} // namespace impl



template <bool Exclusive, typename Float>
struct GradientOptimizer : public impl::GradientOptimizer<GradientOptimizer<Exclusive, Float>, Float>
{
    using impl::GradientOptimizer<GradientOptimizer<Exclusive, Float>, Float>::GradientOptimizer;

    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop && fStop && gStop;
    }
};


template <typename Float>
struct GradientOptimizer<false, Float> : public impl::GradientOptimizer<GradientOptimizer<false, Float>, Float>
{
    using impl::GradientOptimizer<GradientOptimizer<false, Float>, Float>::GradientOptimizer;

    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop || fStop || gStop;
    }
};


namespace poly
{

template <class V = ::nlpp::Vec>
struct GradientOptimizerBase : public ::nlpp::poly::CloneBase<GradientOptimizerBase<V>>
{
    virtual ~GradientOptimizerBase () {}

    using Float = ::nlpp::impl::Scalar<V>;

    virtual void initialize () = 0;

    virtual bool operator () (const nlpp::params::poly::Optimizer_&, const Eigen::Ref<const V>&, Float, const Eigen::Ref<const V>&) = 0;

    virtual int maxIterations () = 0;
};


template <bool Exclusive = true, class V = ::nlpp::Vec>
struct GradientOptimizer : public GradientOptimizerBase<V>,
                           public ::nlpp::stop::GradientOptimizer<Exclusive, ::nlpp::impl::Scalar<V>>
{
    using Float = ::nlpp::impl::Scalar<V>;
    using Impl = ::nlpp::stop::GradientOptimizer<Exclusive, ::nlpp::impl::Scalar<V>>;
    using Impl::Impl;
    using Impl::xTol;
    using Impl::gTol;
    using Impl::fTol;


    virtual void initialize ()
    {
        Impl::initialize();
    }

    virtual bool operator () (const nlpp::params::poly::Optimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return Impl::operator()(optimizer, x, fx, gx);
    }

    virtual int maxIterations () { return Impl::maxIterations(); }


    virtual GradientOptimizer* clone_impl () const { return new GradientOptimizer(*this); }
};


template <class V>
struct GradientOptimizer_ : public ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(Base, ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>);

    GradientOptimizer_ () : Base(std::make_unique<GradientOptimizer<true, V>>()) {}

    using Float = ::nlpp::impl::Scalar<V>;


    void initialized ()
    {
        impl->initialize();
    }

    bool operator () (const nlpp::params::poly::Optimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return impl->operator()(optimizer, x, fx, gx);
    }

    int maxIterations () { return impl->maxIterations(); }
};


} // namespace poly

} // namespace stop

} // namespace nlpp