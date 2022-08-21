#pragma once

#include "helpers/helpers.hpp"
#include "functions_dec.hpp"
// #include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap::impl
{

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
Scalar<V> Functions<Cond, Fs...>::function (const V& x) const
{
    if constexpr(constexpr int id = opId<FuncType_Check, TFs, V>; id >= 0)
        return std::get<id>(fs)(x);

    else if constexpr(constexpr int id = opId<FuncGradType_Check, TFs, V>; id >= 0)
        return std::get<0>(funcGrad(x));

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
Plain<V> Functions<Cond, Fs...>::gradient (const V& x) const
{
    if constexpr(constexpr int id = opId<GradType_0_Check, TFs, V>; id >= 0)
        return std::get<id>(fs)(x);

    else if constexpr(hasOp<GradType_Check, TFs, V> || hasOp<FuncGradType_Check, TFs, V>)
    {
        Plain<V> g(x.rows());
        gradient(x, g);

        return g;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
void Functions<Cond, Fs...>::gradient (const V& x, Plain<V>& g) const
{
    if constexpr(constexpr int id = opId<GradType_1_Check, TFs, V>; id >= 0)
        std::get<id>(fs)(x, g);

    else if constexpr(constexpr int id = opId<GradType_0_Check, TFs, V>; id >= 0)
        g = std::get<id>(fs)(x);

    else if constexpr(hasOp<GradType_Check, TFs, V> || hasOp<FuncGradType_Check, TFs, V>)
        gradient(x, g, function(x));

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
void Functions<Cond, Fs...>::gradient (const V& x, Plain<V>& g, Scalar<V> fx) const
{
    if constexpr(constexpr int id = opId<GradType_2_Check, TFs, V>; id >= 0)
        std::get<id>(fs)(x, g, fx);

    else if constexpr(constexpr int id = opId<GradType_1_Check, TFs, V>; id >= 0)
        std::get<id>(fs)(x, g);

    else if constexpr(constexpr int id = opId<GradType_0_Check, TFs, V>; id >= 0)
        g = std::get<id>(fs)(x);

    else if constexpr(constexpr int id = opId<FuncGradType_Check, TFs, V>; id >= 0)
        funcGrad(x, g);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
std::pair<Scalar<V>, Plain<V>> Functions<Cond, Fs...>::funcGrad (const V& x) const
{
    if constexpr(constexpr int id = opId<FuncGradType_0_Check, TFs, V>; id >= 0)
        return std::get<id>(fs)(x);

    else if constexpr(constexpr int id = opId<FuncGradType_1_Check, TFs, V>; id >= 0)
    {
        Plain<V> g(x.rows());
        Scalar<V> fx = std::get<id>(fs)(x, g);

        return {fx, g};
    }

    else if constexpr(hasOp<FuncType_Check, TFs, V> && hasOp<GradType_Check, TFs, V>)
        return {function(x), gradient(x)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
Scalar<V> Functions<Cond, Fs...>::funcGrad (const V& x, Plain<V>& g) const
{
    if constexpr(constexpr int id = opId<FuncGradType_1_Check, TFs, V>; id >= 0)
        return std::get<id>(fs)(x, g);

    else if constexpr(constexpr int id = opId<FuncGradType_0_Check, TFs, V>; id >= 0)
    {
        Scalar<V> fx;
        std::tie(fx, g) = std::get<id>(fs)(x);

        return fx;
    }

    else if constexpr(hasOp<FuncType_Check, TFs, V> && hasOp<GradType_Check, TFs, V>)
    {
        Scalar<V> fx = function(x);
        gradient(x, g);

        return fx;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}


template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
Plain2D<V> Functions<Cond, Fs...>::hessian (const V& x) const
{
    if constexpr(constexpr int id = opId<HessianType_0_Check, TFs, V>; id >= 0)
        return std::get<id>(fs)(x);

    else if constexpr(constexpr int id = opId<HessianType_1_Check, TFs, V>; id >= 0)
    {
        Plain2D<V> h(x.rows(), x.rows());
        std::get<id>(fs)(x, h);

        return h;
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V, bool Enable> requires Enable
void Functions<Cond, Fs...>::hessian (const V& x, Plain2D<V>& h) const
{
    if constexpr(constexpr int id = opId<HessianType_1_Check, TFs, V>; id >= 0)
        std::get<id>(fs)(x, h);

    else if constexpr(constexpr int id = opId<HessianType_0_Check, TFs, V>; id >= 0)
        h = std::get<id>(fs)(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}



template <Conditions Cond, class... Fs>
template <VecType V1, VecType V2, bool Enable> requires Enable
Scalar<V1> Functions<Cond, Fs...>::gradientDir (const V1& x, const V2& e) const
{
    if constexpr(constexpr int id = opId<GradDirType_0_Check, TFs, V1>; id >= 0)
        return std::get<id>(fs)(x, e);

    else if constexpr(constexpr int id = opId<GradDirType_1_Check, TFs, V1>; id >= 0)
        return std::get<id>(fs)(x, e, function(x));

    else if constexpr(hasOp<GradType_Check, TFs, V1>)
        return gradient(x).dot(e);

    else
        static_assert(always_false<V1>, "The functor has no interface for the given parameter");
}

template <Conditions Cond, class... Fs>
template <VecType V1, VecType V2, bool Enable> requires Enable
Scalar<V1> Functions<Cond, Fs...>::gradientDir (const V1& x, const V2& e, Scalar<V1> fx) const
{
    if constexpr(constexpr int id = opId<GradDirType_1_Check, TFs, V1>; id >= 0)
        return std::get<id>(fs)(x, e, fx);

    else if constexpr(constexpr int id = opId<GradDirType_0_Check, TFs, V1>; id >= 0)
        return std::get<id>(fs)(x, e);

    else if constexpr(hasOp<GradType_Check, TFs, V1>)
    {
        Plain<V1> g(x.rows());
        gradient(x, g, fx);

        return g.dot(e);
    }

    else
        static_assert(always_false<V1>, "The functor has no interface for the given parameter");
}



template <Conditions Cond, class... Fs>
template <VecType V1, VecType V2, bool Enable> requires Enable
Plain<V1> Functions<Cond, Fs...>::hessianDir (const V1& x, const V2& e) const
{
    if constexpr(constexpr int id = opId<HessianDirType_Check, TFs, V1>; id >= 0)
        return std::get<id>(fs)(x, e);

    else if constexpr(hasOp<HessianType_Check, TFs, V1>)
        return hessian(x) * e;

    else
        static_assert(always_false<V1>, "The functor has no interface for the given parameter");
}


// template <Conditions Cond, class... Fs>
// template <class V, bool Enable, std::enable_if_t<Enable, int>>
// Plain2D<V> Functions<Cond, Fs...>::hessianFromDirectional (const Eigen::MatrixBase<V>& x) const
// {
//     Plain2D<V> h(x.rows(), x.rows());
//     hessianFromDirectional(x, h);
//     return h;
// }

// template <Conditions Cond, class... Fs>
// template <class V, bool Enable, std::enable_if_t<Enable, int>>
// void Functions<Cond, Fs...>::hessianFromDirectional (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
// {
//     Plain<V> e = Plain<V>::Constant(x.rows(), Scalar<V>{0.0});
    
//     for(int i = 0; i < e.rows(); ++i)
//     {
//         e(i) = Scalar<V>{1.0};
//         h.row(i) = impl.hessian(x, e);
//         e(i) = Scalar<V>{0.0};
//     }
// }


} // namespace nlpp::wrap::impl