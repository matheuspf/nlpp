#pragma once

#include "Helpers.h"



namespace nlpp::out
{

template <typename Float>
struct Optimizer<0, Float>
{
    // Optimizer (const handy::Print& printer = handy::Print()) : printer(printer) {}

    void initialize () 
    {
    }

    template <typename... Args>
    void operator() (Args&&...)
    {
    }

    // handy::Print printer;
};

template <typename Float>
struct GradientOptimizer<0, Float> : public Optimizer<0, Float> {};


template <typename Float>
struct Optimizer<1, Float> : public Optimizer<0, Float>
{
    Optimizer (const handy::Print& printer = handy::Print("", "\n\n")) : Optimizer<0, Float>(printer) {}

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx)
    {
        printer("x:", x.transpose(), "\nfx:", fx);
    }
};

template <typename Float>
struct GradientOptimizer<1, Float> : public Optimizer<1, Float>
{
    using Base = Optimizer<1, Float>;

    GradientOptimizer (const handy::Print& printer = handy::Print("", "")) : Base(printer) {}

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx, const Eigen::MatrixBase<V>& gx)
    {
        Base::operator()(optimizer, x, fx);
        printer( "\ngx:", gx.transpose(), "\n\n");
    }
};

template <class Float>
struct Optimizer<2, Float>
{
    void initialize () 
    {
        vFx.clear();
        vX.clear();
    }

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx)
    {
        vFx.push_back(fx);
        vX.push_back(::nlpp::impl::cast<Float>(x));
    }

    std::vector<VecX<Float>> vX;
    std::vector<Float> vFx;
};


template <class Float>
struct GradientOptimizer<2, Float> : public Optimizer<2, Float>
{
    using Base = Optimizer<2, Float>;

    void initialize () 
    {
        Base::initialize();
        vGx.clear();
    }

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx, const Eigen::MatrixBase<V>& gx)
    {
        Base::operator()(optimizer, x, fx);
        vGx.push_back(::nlpp::impl::cast<Float>(gx));
    }

    std::vector<VecX<Float>> vGx;
};

} // namespace nlpp::out


namespace nlpp::poly::out
{

template <class V = ::nlpp::Vec>
struct OptimizerBase : public ::nlpp::poly::CloneBase<OptimizerBase<V>>
{
    virtual ~OptimizerBase () {}

    virtual void initialize () = 0;

    virtual void operator() (const ::nlpp::poly::Optimizer<V>&, const V&, ::nlpp::impl::Scalar<::nlpp::impl::Scalar<V>>) = 0;
};

template <class V = ::nlpp::Vec>
struct GradientOptimizerBase : public ::nlpp::poly::CloneBase<GradientOptimizerBase<V>>
{
    virtual ~GradientOptimizerBase () {}

    virtual void initialize () = 0;

    virtual void operator() (const ::nlpp::poly::GradientOptimizer<V>&, const V&, ::nlpp::impl::Scalar<V>, const V&) = 0;
};


template <int Level = 0, class V = ::nlpp::Vec>
struct Optimizer : public OptimizerBase<V>,
                   public ::nlpp::out::Optimizer<Level, ::nlpp::impl::Scalar<V>>
{
    using Impl = ::nlpp::out::Optimizer<Level, ::nlpp::impl::Scalar<V>>;

    virtual void initialize ()
    {
        Impl::initialize();
    }

    virtual void operator() (const ::nlpp::poly::Optimizer<V>& optimizer, const V& x, ::nlpp::impl::Scalar<V> fx)
    {
        return Impl::operator()(optimizer, x, fx);
    }

    virtual Optimizer* clone_impl () const { return new Optimizer(*this); }
};

template <int Level = 0, class V = ::nlpp::Vec>
struct GradientOptimizer : public GradientOptimizerBase<V>,
                           public ::nlpp::out::GradientOptimizer<Level, ::nlpp::impl::Scalar<V>>
{
    using Impl = ::nlpp::out::GradientOptimizer<Level, ::nlpp::impl::Scalar<V>>;

    virtual void initialize ()
    {
        Impl::initialize();
    }

    virtual void operator() (const ::nlpp::poly::GradientOptimizer<V>& optimizer, const V& x, ::nlpp::impl::Scalar<V> fx, const V& gx)
    {
        return Impl::operator()(optimizer, x, fx, gx);
    }

    virtual GradientOptimizer* clone_impl () const { return new GradientOptimizer(*this); }
};


template <class V>
struct Optimizer_ : public ::nlpp::poly::PolyClass<OptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(Optimizer_, Base, ::nlpp::poly::PolyClass<OptimizerBase<V>>);

    Optimizer_ () : Base(std::make_unique<Optimizer<0, V>>()) {}

    void initialize ()
    {
        impl->initialize();
    }

    virtual void operator() (const ::nlpp::poly::Optimizer<V>& optimizer, const V& x, ::nlpp::impl::Scalar<V> fx, const V& gx)
    {
        impl->operator()(optimizer, x, fx, gx);
    }
};



// enum Outputs { QUIET, COMPLETE, STORE };

// static constexpr std::array<const char*, 3> outputNames = { "quiet", "complete", "store" };


template <class V>
struct GradientOptimizer_ : public ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(GradientOptimizer_, Base, ::nlpp::poly::PolyClass<GradientOptimizerBase<V>>);

    GradientOptimizer_ () : Base(std::make_unique<GradientOptimizer<0, V>>()) {}

    void initialize ()
    {
        impl->initialize();
    }

    virtual void operator() (const ::nlpp::poly::GradientOptimizer<V>& optimizer, const V& x, ::nlpp::impl::Scalar<V> fx, const V& gx)
    {
        impl->operator()(optimizer, x, fx, gx);
    }


    // void set(Outputs output)
    // {
    //     switch(output)
    //     {
    //         case QUIET:    impl = std::make_unique<GradientOptimizer<0, V>>(); break;
    //         case COMPLETE: impl = std::make_unique<GradientOptimizer<1, V>>(); break;
    //         case STORE:    impl = std::make_unique<GradientOptimizer<2, V>>(); break;
    //     }
    // }

    // void set (std::string output)
    // {
    //     set(Outputs(handy::find(outputNames, handy::transform(output, output, ::tolower)) - std::begin(outputNames)));
    // }

    // void set (int output)
    // {
    //     set(Outputs(output));
    // }
};

} // namespace nlpp::poly ::out