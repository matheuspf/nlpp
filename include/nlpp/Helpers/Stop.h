#pragma once

#include "Helpers.h"


namespace nlpp
{

namespace stop
{

template <bool Exclusive = false, typename Float = types::Float>
struct GradientOptimizer
{
    GradientOptimizer(int maxIterations_ = 1000, double xTol = 1e-4, double fTol = 1e-4) : 
                      maxIterations_(maxIterations_), xTol(xTol), fTol(fTol), initialized(false) {}

    void initialize ()
    {
        initialized = false;
    }

    template <class Stop, class Output, class V>
    bool operator () (const params::Optimizer<Stop, Output>& optimizer, const Eigen::MatrixBase<V>& x, impl::Scalar<V> fx) 
    {
        bool doStop = false;

        if(initialized)
        {
            bool fStop = std::abs(fx - fx0) < fTol;
            bool xStop = (::nlpp::impl::cast<Float>(x) - x0).norm() < xTol;

            doStop = stop(xStop, fStop);
        }

        fx0 = fx;
        x0 = ::nlpp::impl::cast<Float>(x);
        initialized = true;

        return doStop;
    }


    int maxIterations ()
    {
        return maxIterations_;
    }


    template <typename... Conds>
    bool stop (Conds... conds)
    {
        if constexpr(Exclusive)
            return (conds && ...);

        else
            return (conds || ...);
    }


    Float fx0;

    VecX<Float> x0;


    int maxIterations_;      ///< Maximum number of outer iterations

	Float xTol;            ///< Minimum tolerance on the norm of the input (@c x) between iterations

	Float fTol;            ///< Minimum tolerance on the value of the function (@c x) between iterations

    bool initialized;
};


template <bool Exclusive = false, typename Float = types::Float>
struct GradientOptimizer : public Optimizer<Exclusive, Float>
{
    using Base = Optimizer<Exclusive, Float>;

    GradientOptimizer(int maxIterations_ = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) : 
                      Base(maxIterations_, xTol, fTol), gTol(gTol), initialized(false) {}

    template <class Stop, class Output, class V>
    bool operator () (const GradientOptimizer<Stop, Output>& optimizer,
                      const Eigen::MatrixBase<V>& x, impl::Scalar<V> fx, const Eigen::MatrixBase<V>& gx) 
    {
        bool gStop = gx.norm() < gTol;

        gx0 = ::nlpp::impl::cast<Float>(gx);

        return Base::stop(Base::operator()(optimizer, x, fx), gStop);
    }


    VecX<Float> gx0;

    Float gTol;            ///< Minimum tolerance on the norm of the gradient (@c g) between iterations
};




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


template <typename Float = types::Float>
struct GradientNorm
{
    GradientNorm (int maxIterations_ = 1e3, Float norm = 1e-4) : maxIterations_(maxIterations_), norm(norm)
    {
    }

    void initialize ()
    {
    }


    template <class Stop, class Output, class V>
    bool operator () (const params::Optimizer<Stop, Output>& optimizer, const Eigen::MatrixBase<V>&, double, const Eigen::MatrixBase<V>& gx) 
    {
        return (gx.norm() / gx.size()) < norm;
    }


    int maxIterations () { return maxIterations_; }


    int maxIterations_;
    Float norm;
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


template <class V = ::nlpp::Vec>
struct GradientNorm : public GradientOptimizerBase<V>,
                      public ::nlpp::stop::GradientNorm<::nlpp::impl::Scalar<V>>
{
    using Float = ::nlpp::impl::Scalar<V>;
    using Impl = ::nlpp::stop::GradientNorm<Float>;
    using Impl::Impl;


    virtual void initialize ()
    {
        Impl::initialize();
    }

    virtual bool operator () (const nlpp::params::poly::Optimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return Impl::operator()(optimizer, x, fx, gx);
    }

    virtual int maxIterations () { return Impl::maxIterations(); }

    virtual GradientNorm* clone_impl () const { return new GradientNorm(*this); }
};



enum Stops { IMPROVEMENT_ANY, IMPROVEMENT_ALL, GRADIENT_NORM };

static constexpr std::array<const char*, 3> stopNames = { "improvement_any", "improvement_all" "gradient_norm" };


template <class V>
struct GradientOptimizer_ : public ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(GradientOptimizer_, Base, ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>);
    using Float = ::nlpp::impl::Scalar<V>;

    GradientOptimizer_ () : Base(std::make_unique<GradientOptimizer<true, V>>()) {}


    void initialize ()
    {
        impl->initialize();
    }

    bool operator () (const nlpp::params::poly::Optimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return impl->operator()(optimizer, x, fx, gx);
    }

    int maxIterations () { return impl->maxIterations(); }


    void set (Stops stop)
    {
        switch(stop)
        {
            case IMPROVEMENT_ANY: impl = std::make_unique<GradientOptimizer<false, V>>(); break;
            case IMPROVEMENT_ALL: impl = std::make_unique<GradientOptimizer<true, V>>();  break;
            case GRADIENT_NORM:   impl = std::make_unique<GradientNorm<V>>();             break;
        }
    }

    void set (std::string stop)
    {
        set(Stops(handy::find(stopNames, handy::transform(stop, stop, ::tolower)) - std::begin(stopNames)));
    }

    void set (int stop)
    {
        set(Stops(stop));
    }
};


} // namespace poly

} // namespace stop

} // namespace nlpp