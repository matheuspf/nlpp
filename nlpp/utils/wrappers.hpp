#pragma once

#include "helpers/helpers.hpp"
#include "wrappers_dec.hpp"
// #include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap::impl
{

// template <class Impl>
// Function<Impl>::Function (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::function (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<V>::Function)
        return impl.function(x);
    
    else if constexpr(HasOp<V>::FuncGrad)
        return impl.getFuncGrad(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}


template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
{
    if constexpr(HasOp<V>::Gradient_2)
        impl.gradient(x, g, fx);

    else if constexpr(HasOp<V>::Gradient_0)
        impl.gradient(x, g);

    else if constexpr(HasOp<V>::Gradient_1)
        g = impl.gradient(x);

    else if constexpr(HasOp<V>::FuncGrad)
        impl.getFuncGrad(x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
{
    if constexpr(HasOp<V>::Gradient_0)
        impl.gradient(x, g);

    else if constexpr(HasOp<V>::Gradient_1)
        g = impl.gradient(x);

    else if constexpr(HasOp<V>::Gradient_2)
        impl.gradient(x, g, std::nan("0"));

    else if constexpr(HasOp<V>::FuncGrad)
        impl.getFuncGrad(x, g, true);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Plain<V> Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<V>::Gradient_0 || HasOp<V>::Gradient_2 || HasOp<V>::FuncGrad)
    {
        Plain<V> g(x.rows());

        if constexpr(HasOp<V>::Gradient_0)
            impl.gradient(x, g);
        
        else if constexpr(HasOp<V>::Gradient_2)
            impl.gradient(x, g, std::nan("0"));

        else
            impl.getFuncGrad(x, g, true);

        return g;
    }

    else if constexpr(HasOp<V>::Gradient_1)
        return impl.gradient(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Scalar<V> Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
{
    if constexpr(HasOp<V>::Directional)
        return impl.gradient(x, e);

    else
        return gradient(x).dot(e);
}


template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::funcGrad (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad) const
{
    if constexpr(HasOp<V>::FuncGrad)
        return impl.getFuncGrad(x, g, calcGrad);

    else if constexpr(HasOp<V>::Function && HasOp<V>::Gradient)
    {
        auto f = function(x);

        if(calcGrad)
            gradient(x, g, f);

        return f;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Impl>::funcGrad (const Eigen::MatrixBase<V>& x) const
{
    if constexpr(HasOp<V>::FuncGrad_0 || HasOp<V>::FuncGrad_1)
    {
        Plain<V> g(x.rows());
        Scalar<V> fx = impl.funcGrad(x, g);

        return {fx, g};
    } 
 
    else if constexpr(HasOp<V>::FuncGrad_2)
        return impl.funcGrad(x);

    else if constexpr(HasOp<V>::Function && HasOp<V>::Gradient)
        return {function(x), gradient(x)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
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