#pragma once

#include "line_search/line_search.hpp"


namespace nlpp::ls
{

namespace impl
{

/** @brief Backtracking line search
 * 
 *  @details Search for a point @c a satisfying the first of the Wolfe conditions given the function/gradient
 *           functor f
*/
template <class Base_>
struct Backtracking : public Base_
{
    NLPP_USING_LINESEARCH(Base, Base_);

    /// Some reasonable default values
    Backtracking (const InitialStep &initialStep = InitialStep(1.0), Float c = 1e-4, Float rho = 0.5, Float aMin = constants::eps) : Base(initialStep), c(c), rho(rho), aMin(aMin)
    {
        assert(c < 1.0 && "c must be smaller than 1.0");
        assert(rho < 1.0 && "rho must be smaller than 1.0");
    }

    /** @brief The line search procedure
     *  @param f A function/gradient functor, projected on a single dimension
    */
    template <class Function>
    Float lineSearch (const Function& f)
    {
        auto [f0, g0] = f(0.0);
        auto a = initialStep(*this, f0, g0);

        ///< Check Wolfe's first condition @f$f(x + a * p) \leq f(x) + c_1 * a * p \intercall \nabla f(x)@f$
        while (a > aMin && f.function(a) > f0 + c * a * g0)
            a = rho * a;

        return a;
    }

    Float c; ///< Factor to control the linear Wolfe condition (@c c1)

    Float aMin; ///< Smallest step acceptable

    Float rho; ///< Factor to reduce @c a
};

} // namespace impl


template <typename Float_ = types::Float, class InitialStep_ = ConstantStep<Float_>>
struct Backtracking : public impl::Backtracking<LineSearch<Backtracking<Float_, InitialStep_>>>
{
    NLPP_USING_LINESEARCH(Base, impl::Backtracking<LineSearch<Backtracking<Float_, InitialStep_>>>);
};

} // namespace nlpp::ls


namespace nlpp::traits
{

template <typename Float_, class InitialStep_>
struct LineSearch<nlpp::ls::Backtracking<Float_, InitialStep_>>
{
    using Float = Float_;
    using InitialStep = InitialStep_;
};

} // namespace nlpp::traits
