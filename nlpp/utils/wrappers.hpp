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
    static_assert(isFunction<Impl, V>, "The functor has no interface for the given parameter");

    return Impl::operator()(x);
}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::operator () (const Eigen::MatrixBase<V>& x)
{
    return function(x);
}


template <class Impl>
Gradient<Impl>::Gradient (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx)
{
    if constexpr(isGradient_2<Impl, V>)
        Impl::operator()(x, g, fx);

    else if constexpr(isGradient_0<Impl, V>)
        Impl::operator()(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    if constexpr(isGradient_0<Impl, V>)
        Impl::operator()(x, g);

    else if constexpr(isGradient_1<Impl, V>)
        g = Impl::operator()(x);

    else if constexpr(isGradient_2<Impl, V>)
        Impl::operator()(x, g, std::nan("0"));

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Plain<V> Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isGradient_0<Impl, V> || isGradient_2<Impl, V>)
    {
        Plain<V> g(x.rows());

        if constexpr(isGradient_0<Impl, V>)
            Impl::operator()(x, g);
        
        else
            Impl::operator()(x, g, std::nan("0"));

        return g;
    }

    else if constexpr(isGradient_1<Impl, V>)
        return Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Scalar<V> Gradient<Impl>::directional (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e)
{
    if constexpr(isDirectional<Impl, V>)
        return Impl::operator()(x, e);

    else
        return gradient(x) * e;
}

template <class Impl>
template <class V>
void Gradient<Impl>::operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    gradient(x, g);
}

template <class Impl>
template <class V>
Plain<V> Gradient<Impl>::operator() (const Eigen::MatrixBase<V>& x)
{
    return gradient(x);
}


template <class Func, class Grad>
template <class F, class G, std::enable_if_t<handy::HasConstructor<G>::value, int>>
FunctionGradient<Func, Grad>::FunctionGradient (const F& f, const G& g) : Function<Func>(f), Gradient<Grad>(g) {}

template <class Func, class Grad>
template <class F, class G, std::enable_if_t<!handy::HasConstructor<G>::value, int>>
FunctionGradient<Func, Grad>::FunctionGradient (const F& f, const G& g) : Function<Func>(f), Gradient<Grad>(g) {}

template <class Func, class Grad>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, Grad>::functionGradient (const Eigen::MatrixBase<V>& x)
{
    auto f = function(x);
    return {f, gradient(x, f)};
}

template <class Func, class Grad>
template <class V>
Scalar<V> FunctionGradient<Func, Grad>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    auto f = function(x);

    if(calcGrad)
        gradient(x, g, f);

    return f;
}

template <class Func, class Grad>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, Grad>::operator() (const Eigen::MatrixBase<V>& x)
{
    return functionGradient(x);
}

template <class Func, class Grad>
template <class V>
Scalar<V> FunctionGradient<Func, Grad>::operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    return functionGradient(x, g, calcGrad);
}


template <class Impl>
FunctionGradient<Impl>::FunctionGradient (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    if constexpr(isFuncGrad_0<Impl, V>)
        return Impl::operator()(x, g, calcGrad);

    else if constexpr(isFuncGrad_1<Impl, V>)
        return Impl::operator()(x, g);

    else if constexpr(isFuncGrad_2<Impl, V>)
    {
        Scalar<V> f;
        std::tie(f, g) = Impl::operator()(x);
        return f;
    }

    else if constexpr(isFunction<Impl, V> && (isGradient_0<Impl, V> || isGradient_1<Impl, V>))
    {
        auto f = Function<Impl>(*this)(x);

        if(calcGrad)
            Gradient<Impl>(*this)(x, g, f);

        return f;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x)
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
        return {Function<Impl>(*this)(x), Gradient<Impl>(*this)(x)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::function (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isFunction<Impl, V>)
        return Function<Impl>(*this)(x);

    else
        return functionGrad(x);
}

template <class Impl>
template <class V>
void FunctionGradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    if constexpr(isGradient_0<Impl, V> || isGradient_1<Impl, V>)
        Gradient<Impl>(*this)(x, g);

    else
        functionGrad(x);
}

template <class Impl>
template <class V>
Plain<V> FunctionGradient<Impl>::gradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isGradient_0<Impl, V> || isGradient_1<Impl, V>)
        return Gradient<Impl>(*this)(x);

    else
        return functionGrad(x);
}


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

template <class Impl>
template <class V, class U>
Plain<V> Hessian<Impl>::operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
{
    return hessian(x, e);
}

template <class Impl>
template <class V>
Plain2D<V> Hessian<Impl>::operator() (const Eigen::MatrixBase<V>& x)
{
    return hessian(x);
}

} // namespace nlpp::wrap::impl