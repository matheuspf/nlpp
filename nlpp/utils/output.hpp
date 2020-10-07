#pragma once

#include "helpers/helpers_dec.hpp"


namespace nlpp::out
{

template <int Level = 0, typename Float = types::Float>
struct Optimizer;

template <int Level = 0, typename Float = types::Float>
struct GradientOptimizer;


template <typename Float>
struct Optimizer<0, Float>
{
    template <typename... Args>
    void operator() (Args&&...)
    {
    }
};

template <typename Float>
struct GradientOptimizer<0, Float> : public Optimizer<0, Float> {};


template <typename Float>
struct Optimizer<1, Float> : public Optimizer<0, Float>
{
    Optimizer (const handy::Print& printer = handy::Print("", "\n\n")) : printer(printer) {}

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx)
    {
        printer("x:", x.transpose(), "\nfx:", fx);
    }

    handy::Print printer;
};

template <typename Float>
struct GradientOptimizer<1, Float> : public Optimizer<1, Float>
{
    using Base = Optimizer<1, Float>;
    using Base::printer;

    GradientOptimizer (const handy::Print& printer = handy::Print("", "\n\n")) : Base(printer) {}

    template <class Opt, class V, class U>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx, const Eigen::MatrixBase<U>& gx)
    {
        printer("x:", x.transpose(), "\nfx:", fx, "\ngx:", gx.transpose());
    }
};

template <class Float>
struct Optimizer<2, Float>
{
    Optimizer(std::vector<VecX<Float>>& vx, std::vector<Float>& vfx) : vx(&vx), vfx(&vfx) {}
    Optimizer(std::vector<VecX<Float>>* vx = nullptr, std::vector<Float>* vfx = nullptr) : vx(vx), vfx(vfx) {}

    ~Optimizer() = default; // no ownership

    template <class Opt, class V>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx)
    {
        vx->push_back(::nlpp::impl::cast<Float>(x));
        vfx->push_back(fx);
    }

    std::vector<VecX<Float>>* vx;
    std::vector<Float>* vfx;
};


template <class Float>
struct GradientOptimizer<2, Float> : public Optimizer<2, Float>
{
    using Base = Optimizer<2, Float>;

    GradientOptimizer(std::vector<VecX<Float>>& vx,
                      std::vector<Float>& vfx,
                      std::vector<VecX<Float>>& vgx) : Base(vx, vfx), vgx(&vgx) {}
    GradientOptimizer(std::vector<VecX<Float>>* vx = nullptr,
                      std::vector<Float>* vfx = nullptr,
                      std::vector<VecX<Float>>* vgx = nullptr) : Base(vx, vfx), vgx(vgx) {}

    template <class Opt, class V, class U>
    void operator() (const Opt& optimizer, const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx, const Eigen::MatrixBase<U>& gx)
    {
        Base::operator()(optimizer, x, fx);
        vgx->push_back(::nlpp::impl::cast<Float>(gx));
    }

    std::vector<VecX<Float>>* vgx;
};

} // namespace nlpp::out


namespace nlpp::poly::out
{

template <class V = ::nlpp::Vec>
struct OptimizerBase : public ::nlpp::poly::CloneBase<OptimizerBase<V>>
{
    virtual ~OptimizerBase () {}

    virtual void initialize () = 0;

    virtual void operator() (const ::nlpp::poly::Optimizer<V>&, const V&, ::nlpp::impl::Scalar<V>) = 0;
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


template <class V = ::nlpp::Vec>
struct Optimizer_ : public ::nlpp::poly::PolyClass<OptimizerBase<V>>
{
    NLPP_USING_POLY_CLASS(Optimizer_, Base, ::nlpp::poly::PolyClass<OptimizerBase<V>>);

    Optimizer_ () : Base(std::make_unique<Optimizer<0, V>>()) {}

    void initialize ()
    {
        impl->initialize();
    }

    virtual void operator() (const ::nlpp::poly::Optimizer<V>& optimizer, const V& x, ::nlpp::impl::Scalar<V> fx)
    {
        impl->operator()(optimizer, x, fx);
    }
};



// enum Outputs { QUIET, COMPLETE, STORE };

// static constexpr std::array<const char*, 3> outputNames = { "quiet", "complete", "store" };


template <class V = ::nlpp::Vec>
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