#pragma once

#include "line_search/line_search.hpp"
#include "line_search/initial_step/constant.hpp"

namespace nlpp::ls
{

namespace impl
{

template <class Base_>
struct Goldstein : public Base_
{
    NLPP_USING_LINESEARCH(Base, Base_);

    Goldstein(Float c = 0.2, Float rho1 = 0.5, Float rho2 = 1.5, const InitialStep &initialStep = InitialStep(),
              Float aMin = constants::eps, int maxIter = 100) :
              mu1(c), mu2(1.0 - c), Base(initialStep), rho1(rho1), rho2(rho2), aMin(aMin), maxIter(maxIter)
    {
        assert(c < 0.5 && "c must be smaller than 0.5");
        assert(rho1 < 1.0 && "rho1 must be smaller than 1.0");
        assert(rho2 > 1.0 && "rho2 must be greater than 1.0");
    }

    template <class Function>
    Float lineSearch (const Function& f)
    {
        auto [f0, g0] = f(0.0);

        Float a0 = initialStep(*this, f0, g0);
        Float a = a0, safeGuard = a0;

        int iter = 0;

        while (a > aMin && ++iter < maxIter)
        {
            Float fa, ga;

            std::tie(fa, ga) = f(a);

            if (fa > f0 + mu1 * a * g0)
            {
                a = a * rho1;
                continue;
            }

            safeGuard = a;

            if (fa < f0 + mu2 * a * g0)
            {
                a = a * rho2;

                continue;
            }

            break;
        }

        return iter < maxIter ? a : safeGuard;
    }

    Float mu1, mu2;
    Float rho1, rho2;
    Float aMin;
    int maxIter;
};

} // namespace impl

template <typename Float_ = types::Float, class InitialStep_ = ConstantStep<Float_>>
struct Goldstein : public impl::Goldstein<LineSearch<Goldstein<Float_, InitialStep_>>>
{
    NLPP_USING_LINESEARCH(Base, impl::Goldstein<LineSearch<Goldstein<Float_, InitialStep_>>>);
};

} // namespace nlpp::ls


namespace nlpp::traits
{

template <typename Float_, class InitialStep_>
struct LineSearch<nlpp::ls::Goldstein<Float_, InitialStep_>>
{
    using Float = Float_;
    using InitialStep = InitialStep_;
};

} // namespace nlpp::traits



// namespace poly
// {

//     template <class V = Vec, class InitialStep = ConstantStep<typename V::Scalar>>
//     struct Goldstein : public impl::Goldstein<::nlpp::impl::Scalar<V>, InitialStep>,
//                         public LineSearchBase<V>
//     {
//         using Float = ::nlpp::impl::Scalar<V>;
//         using Interface = LineSearch<Float>;
//         using Impl = impl::Goldstein<Float, InitialStep>;
//         using Impl::Impl;

//         void initialize()
//         {
//             Impl::initialize();
//         }

//         Float lineSearch(::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<V>, ::nlpp::Vec> f)
//         {
//             return Impl::lineSearch(f);
//         }

//         virtual Goldstein *clone_impl() const { return new Goldstein(*this); }
//     };

// } // namespace poly

