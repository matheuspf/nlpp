#pragma once

#include "helpers/helpers.hpp"
#include "wrappers.hpp"
// #include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap::impl
{

// template <class Impl>
// Function<Impl>::Function (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::function (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(isFunction<Impl, V>)
        return impl(x);
    
    else if constexpr(isFuncGrad<Impl, V>)
        return getFuncGrad(*this, x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
{
    if constexpr(isGradient_2<Impl, V>)
        impl(x, g, fx);

    else if constexpr(isGradient_0<Impl, V>)
        impl(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = impl(x);

    else if constexpr(isFuncGrad<Impl, V>)
        getFuncGrad(*this, x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
{
    if constexpr(isGradient_0<Impl, V>)
        impl(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = impl(x);

    else if constexpr(isGradient_2<Impl, V>)
        impl(x, g, std::nan("0"));

    else if constexpr(isFuncGrad<Impl, V>)
        getFuncGrad(*this, x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Plain<V> Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(isGradient_0<Impl, V> || isGradient_2<Impl, V> || isFuncGrad<Impl, V>)
    {
        Plain<V> g(x.rows());

        if constexpr(isGradient_0<Impl, V>)
            impl(x, g);
        
        else if constexpr(isGradient_2<Impl, V>)
            impl(x, g, std::nan("0"));

        else
            getFuncGrad(*this, x, g, true);

        return g;
    }

    else if constexpr(isGradient_1<Impl, V>)
        return impl(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Scalar<V> Gradient<Impl>::directional (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
{
    if constexpr(isDirectional<Impl, V>)
        return impl(x, e);

    else
        return gradient(x) * e;
}


template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad) const
{
    if constexpr(isFuncGrad<Impl, V>)
        return getFuncGrad(x, g, calcGrad);

    else if constexpr(isFunction<Impl, V> && (isGradient_0<Impl, V> || isGradient_1<Impl, V>))
    {
        auto f = func(x);

        if(calcGrad)
            grad(x, g, f);

        return f;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(isFuncGrad_0<Impl, V> || isFuncGrad_1<Impl, V>)
    {
        Plain<V> g(x.rows());
        Scalar<V> fx = impl(x, g);

        return {fx, g};
    }

    else if constexpr(isFuncGrad_2<Impl, V>)
        return impl(x);

    else if constexpr(isFunction<Impl, V> && isGradient<Impl, V>)
        return {func(x), grad(x)};
        // return {Impl::function(x), Impl::gradient(x)};

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
Scalar<V> getFuncGrad (const Impl& impl, const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
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


// template <class Impl>
// Hessian<Impl>::Hessian (const Impl& impl) : Impl(impl) {}

// template <class Impl>
// template <class V, class U>
// Plain<V> Hessian<Impl>::hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
// {
//     if constexpr(isHessian_1<Impl, V, U>)
//         return impl(x, e);

//     else if constexpr(isHessian_2<Impl, V>)
//         return impl(x) * e;

//     else
//         static_assert(always_false<V>, "The functor has no interface for the given parameter");
// }

// template <class Impl>
// template <class V>
// Plain2D<V> Hessian<Impl>::hessian (const Eigen::MatrixBase<V>& x)
// {
//     if constexpr(isHessian_1<Impl, V>)
//     {
//         Plain<V> e = Plain<V>::Constant(x.rows(), Scalar<V>{0.0});
//         Plain2D<V> m(x.rows(), x.rows());

//         for(int i = 0; i < e.rows(); ++i)
//         {
//             e(i) = Scalar<V>{1.0};
//             m.row(i) = impl(x, e);
//             e(i) = Scalar<V>{0.0};
//         }

//         return m;
//     }

//     else if constexpr(isHessian_2<Impl, V>)
//         return impl(x);

//     else
//         static_assert(always_false<V>, "The functor has no interface for the given parameter");
// }

} // namespace nlpp::wrap::impl