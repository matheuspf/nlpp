#pragma once

#include "Helpers.h"



namespace nlpp
{

namespace out
{

template <typename Float>
struct GradientOptimizer<0, Float>
{
    void initialize () 
    {
    }

    template <typename... Args>
    void operator() (Args&&...)
    {
    }
};


template <typename Float>
struct GradientOptimizer<1, Float>
{
    void initialize ()
    {
    }

    template <class LineSearch, class Stop, class Output, class V>
    void operator() (const params::LineSearchOptimizer<LineSearch, Stop, Output>& optimizer,
                     const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx)
    {
        handy::print("x:", x.transpose(), "\nfx:", fx, "\ngx:", gx.transpose(), "\n") << std::flush;
    }
};


template <class Float>
struct GradientOptimizer<2, Float>
{
    void initialize () 
    {
        vFx.clear();
        vX.clear();
        vGx.clear();
    }

    template <class LineSearch, class Stop, class Output, class V>
    void operator() (const params::LineSearchOptimizer<LineSearch, Stop, Output>& optimizer,
                     const Eigen::MatrixBase<V>& x, Float fx, const Eigen::MatrixBase<V>& gx)
    {
        vFx.push_back(fx);
        vX.push_back(::nlpp::impl::cast<Float>(x));
        vGx.push_back(::nlpp::impl::cast<Float>(gx));
    }


    std::vector<VecX<Float>> vX;
    std::vector<Float> vFx;
    std::vector<VecX<Float>> vGx;
};



namespace poly
{

template <class V = ::nlpp::Vec>
struct GradientOptimizerBase : public ::nlpp::poly::CloneBase<GradientOptimizerBase<V>>
{
    virtual ~GradientOptimizerBase () {}

    using Float = ::nlpp::impl::Scalar<V>;

    virtual void initialize () = 0;

    virtual void operator() (const nlpp::params::poly::LineSearchOptimizer_&, const Eigen::Ref<const V>&, Float, const Eigen::Ref<const V>&) = 0;
};


template <int Level = 0, class V = ::nlpp::Vec>
struct GradientOptimizer : public GradientOptimizerBase<V>,
                           public ::nlpp::out::GradientOptimizer<Level, ::nlpp::impl::Scalar<V>>
{
    using Float = ::nlpp::impl::Scalar<V>;
    using Impl = ::nlpp::out::GradientOptimizer<Level, ::nlpp::impl::Scalar<V>>;


    virtual void initialize ()
    {
        Impl::initialize();
    }

    virtual void operator() (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        return Impl::operator()(optimizer, x, fx, gx);
    }
    

    virtual GradientOptimizer* clone_impl () const { return new GradientOptimizer(*this); }
};


template <class V>
struct GradientOptimizer_ : public ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(Base, ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>);

    using Float = ::nlpp::impl::Scalar<V>;


    GradientOptimizer_ () : Base(std::make_unique<GradientOptimizer<0, V>>()) {}


    void initialize ()
    {
        impl->initialize();
    }

    void operator () (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        impl->operator()(optimizer, x, fx, gx);
    }
};


} // namespace poly 

} // namespace out

} // namespace nlpp