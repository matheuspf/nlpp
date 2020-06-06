#pragma once

#include "line_search/line_search.hpp"


namespace nlpp::ls
{

namespace impl
{

template <class Base_>
struct ConstantStep : public Base_
{
    NLPP_USING_LINESEARCH(Base, Base_);

    template <class Function>
    Float lineSearch (const Function&)
    {
        return initialStep();
    }
};

} // namespace impl

template <typename Float_ = types::Float, class InitialStep_ = ConstantStep<Float_>>
struct ConstantStep : public impl::ConstantStep<LineSearch<ConstantStep<Float_, InitialStep_>>>
{
    NLPP_USING_LINESEARCH(Base, impl::ConstantStep<LineSearch<ConstantStep<Float_, InitialStep_>>>);
};

} // namespace nlpp::ls

namespace nlpp::traits
{

template <typename Float_, class InitialStep_>
struct LineSearch<nlpp::ls::ConstantStep<Float_, InitialStep_>>
{
    using Float = Float_;
    using InitialStep = InitialStep_;
};

} // namespace nlpp::traits
