#pragma once

#include "helpers/helpers.hpp"
#include "wrappers_dec.hpp"
// #include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap::impl
{

// template <class Impl>
// Function<Impl>::Function (const Impl& impl) : Impl(impl) {}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
Scalar<V> Functions<Cond, Impl>::function (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<IsFunction, V, TFs>)
        return impl.function(x);
    
    else if constexpr(HasOp<IsFuncGrad, V, TFs>)
        return impl.getFuncGrad(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}


template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
void Functions<Cond, Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
{
    if constexpr(HasOp<IsGradient_2, V, TFs>)
        impl.gradient(x, g, fx);

    else if constexpr(HasOp<IsGradient_0, V, TFs>)
        impl.gradient(x, g);

    else if constexpr(HasOp<IsGradient_1, V, TFs>)
        g = impl.gradient(x);

    else if constexpr(HasOp<IsFuncGrad, V, TFs>)
        impl.getFuncGrad(x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
void Functions<Cond, Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
{
    if constexpr(HasOp<IsGradient_0, V, TFs>)
        impl.gradient(x, g);

    else if constexpr(HasOp<IsGradient_1, V, TFs>)
        g = impl.gradient(x);

    else if constexpr(HasOp<IsGradient_2, V, TFs>)
        impl.gradient(x, g, std::nan("0"));

    else if constexpr(HasOp<IsFuncGrad, V, TFs>)
        impl.getFuncGrad(x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
Plain<V> Functions<Cond, Impl>::gradient (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<IsGradient_0, V, TFs> || HasOp<IsGradient_2, V, TFs> || HasOp<IsFuncGrad, V, TFs>)
    {
        Plain<V> g(x.rows());

        if constexpr(HasOp<IsGradient_0, V, TFs>)
            impl.gradient(x, g);
        
        else if constexpr(HasOp<IsGradient_2, V, TFs>)
            impl.gradient(x, g, std::nan("0"));

        else
            impl.getFuncGrad(x, g, true);

        return g;
    }

    else if constexpr(HasOp<IsGradient_1, V, TFs>)
        return impl.gradient(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, class U, bool Enable, std::enable_if_t<Enable, int>>
Scalar<V> Functions<Cond, Impl>::gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
{
    if constexpr(HasOp<IsDirectional, V, TFs>)
        return impl.gradient(x, e);

    else
        return gradient(x).dot(e);
}


template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
Scalar<V> Functions<Cond, Impl>::funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad) const
{
    if constexpr(HasOp<IsFuncGrad, V, TFs>)
        return impl.getFuncGrad(x, g, calcGrad);

    else if constexpr(HasOp<IsFunction, V, TFs> && HasOp<IsGradient, V, TFs>)
    {
        auto f = function(x);

        if(calcGrad)
            gradient(x, g, f);

        return f;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
std::pair<Scalar<V>, Plain<V>> Functions<Cond, Impl>::funcGrad (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<IsFuncGrad_0, V, TFs> || HasOp<IsFuncGrad_1, V, TFs>)
    {
        Plain<V> g(x.rows());
        Scalar<V> fx = impl.funcGrad(x, g);

        return {fx, g};
    } 
 
    else if constexpr(HasOp<IsFuncGrad_2, V, TFs>)
        return impl.funcGrad(x);

    else if constexpr(HasOp<IsFunction, V, TFs> && HasOp<IsGradient, V, TFs>)
        return {function(x), gradient(x)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}



template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
void Functions<Cond, Impl>::hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
{
    if constexpr(HasOp<IsHessian_0, V, TFs>)
        impl.hessian(x, h);

    else if constexpr(HasOp<IsHessian_1, V, TFs>)
        h = impl.hessian(x);

    else if constexpr(HasOp<IsHessian_2, V, TFs>)
        hessianFromDirectional(x, h);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
Plain2D<V> Functions<Cond, Impl>::hessian (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<IsHessian_1, V, TFs>)
        return impl.hessian(x);

    else if constexpr(HasOp<IsHessian_0, V, TFs>)
    {
        Plain2D<V> h(x.rows(), x.rows());
        impl.hessian(x, h);
        return h;
    }

    else if constexpr(HasOp<IsHessian_2, V, TFs>)
        return hessianFromDirectional(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, class U, bool Enable, std::enable_if_t<Enable, int>>
Plain<V> Functions<Cond, Impl>::hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
{
    if constexpr(HasOp<IsHessian_2, V, TFs>)
        return impl.hessian(x, e);

    else if constexpr(HasOp<IsHessian_0, V, TFs>)
    {
        Plain2D<V> h(x.rows(), x.rows());
        impl.hessian(x, h);
        return h * x;
    }

    else if constexpr(HasOp<IsHessian_1, V, TFs>)
        return impl.hessian(x) * x;

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
Plain2D<V> Functions<Cond, Impl>::hessianFromDirectional (const Eigen::MatrixBase<V>& x) const
{
    Plain2D<V> h(x.rows(), x.rows());
    hessianFromDirectional(x, h);
    return h;
}

template <Conditions Cond, class Impl>
template <class V, bool Enable, std::enable_if_t<Enable, int>>
void Functions<Cond, Impl>::hessianFromDirectional (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
{
    Plain<V> e = Plain<V>::Constant(x.rows(), Scalar<V>{0.0});
    
    for(int i = 0; i < e.rows(); ++i)
    {
        e(i) = Scalar<V>{1.0};
        h.row(i) = impl.hessian(x, e);
        e(i) = Scalar<V>{0.0};
    }
}


} // namespace nlpp::wrap::impl