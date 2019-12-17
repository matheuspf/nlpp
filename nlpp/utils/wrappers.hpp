#pragma once

#include "helpers/helpers.hpp"
#include "wrappers.hpp"
// #include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap::impl
{

template <class Impl>
Function<Impl>::Function (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::function (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isFunction<Impl, V>)
        return Impl::operator()(x);
    
    else if constexpr(isFuncGrad<Impl, V>)
        return getFuncGrad(*this, x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");

}

template <class... _Impl>
template <class V>
void Gradient<_Impl...>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx)
{
    if constexpr(isGradient_2<Impl, V>)
        Impl::operator()(x, g, fx);

    else if constexpr(isGradient_0<Impl, V>)
        Impl::operator()(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = Impl::operator()(x);

    else if constexpr(isFuncGrad<Impl, V>)
        getFuncGrad(*this, x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class... _Impl>
template <class V>
void Gradient<_Impl...>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    if constexpr(isGradient_0<Impl, V>)
        Impl::operator()(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = Impl::operator()(x);

    else if constexpr(isGradient_2<Impl, V>)
        Impl::operator()(x, g, std::nan("0"));

    else if constexpr(isFuncGrad<Impl, V>)
        getFuncGrad(*this, x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class... _Impl>
template <class V>
Plain<V> Gradient<_Impl...>::gradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isGradient_0<Impl, V> || isGradient_2<Impl, V> || isFuncGrad<Impl, V>)
    {
        Plain<V> g(x.rows());

        if constexpr(isGradient_0<Impl, V>)
            Impl::operator()(x, g);
        
        else if constexpr(isGradient_2<Impl, V>)
            Impl::operator()(x, g, std::nan("0"));

        else
            getFuncGrad(*this, x, g, true);

        return g;
    }

    else if constexpr(isGradient_1<Impl, V>)
        return Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class... _Impl>
template <class V>
Scalar<V> Gradient<_Impl...>::directional (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e)
{
    if constexpr(isDirectional<Impl, V>)
        return Impl::operator()(x, e);

    else
        return gradient(x) * e;
}


template <class... _Impl>
template <class V>
Scalar<V> FunctionGradient<_Impl...>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    if constexpr(isFuncGrad<Impl, V>)
        return getFuncGrad(x, g, calcGrad);

    else if constexpr(isFunction<Impl, V> && (isGradient_0<Impl, V> || isGradient_1<Impl, V>))
    {
        auto f = Func::operator()(*this)(x);

        if(calcGrad)
            Grad::operator()(x, g, f);

        return f;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class... _Impl>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<_Impl...>::functionGradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isFuncGrad_0<Impl, V> || isFuncGrad_1<Impl, V>)
    {
        Plain<V> g(x.rows());
        Scalar<V> fx = Impl::operator()(x, g);

        return {fx, g};
    }

    else if constexpr(isFuncGrad_2<Impl, V>)
        return Impl::operator()(x);

    else if constexpr(isFunction<Impl, V> && (isGradient_0<Impl, V> || isGradient_1<Impl, V>))
        return {Func::operator()(x), Grad::operator()(x)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl, class V>
Scalar<V> getFuncGrad (Impl& impl, const Eigen::MatrixBase<V>& x)
{
    Plain<V> g;
    return getFuncGrad(impl, x, g, false);
}

template <class Impl, class V>
Scalar<V> getFuncGrad (Impl& impl, const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    if constexpr(isFuncGrad_0<Impl, V>)
        return impl(x, g, calcGrad);

    else if constexpr(isFuncGrad_1<Impl, V>)
        return impl(x, g);

    else if constexpr(isFuncGrad_2<Impl, V>)
    {
        Scalar<V> f;
        std::tie(f, g) = impl(x);
        return f;
    }
}

// template <class... _Impl>
// template <class V>
// Scalar<V> FunctionGradient<_Impl...>::function (const Eigen::MatrixBase<V>& x)
// {
//     return Func::operator()(x);

//     if constexpr(isFunction<Impl, V>)
//         return Func::operator()(x);

//     else
//         return functionGrad(x);
// }

// template <class... _Impl>
// template <class V>
// void FunctionGradient<_Impl...>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
// {
//     if constexpr(isGradient_0<Impl, V> || isGradient_1<Impl, V>)
//         Grad::operator()(x, g);

//     else
//         functionGrad(x);
// }

// template <class... _Impl>
// template <class V>
// Plain<V> FunctionGradient<_Impl...>::gradient (const Eigen::MatrixBase<V>& x)
// {
//     if constexpr(isGradient_0<Impl, V> || isGradient_1<Impl, V>)
//         return Grad::operator()(x);

//     else
//         return functionGrad(x);
// }


template <class Impl>
Hessian<Impl>::Hessian (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V, class U>
Plain<V> Hessian<Impl>::hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
{
    if constexpr(isHessian_1<Impl, V, U>)
        return Impl::operator()(x, e);

    else if constexpr(isHessian_2<Impl, V>)
        return Impl::operator()(x) * e;

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Plain2D<V> Hessian<Impl>::hessian (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isHessian_1<Impl, V>)
    {
        Plain<V> e = Plain<V>::Constant(x.rows(), Scalar<V>{0.0});
        Plain2D<V> m(x.rows(), x.rows());

        for(int i = 0; i < e.rows(); ++i)
        {
            e(i) = Scalar<V>{1.0};
            m.row(i) = Impl::operator()(x, e);
            e(i) = Scalar<V>{0.0};
        }

        return m;
    }

    else if constexpr(isHessian_2<Impl, V>)
        return Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

} // namespace nlpp::wrap::impl