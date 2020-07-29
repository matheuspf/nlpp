#pragma once

#include "../utils/optimizer.hpp"
#include "../quasi_newton/lbfgs/lbfgs_dec.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct HyperbolicPenalty : public Base_
{
    NLPP_USING_OPTIMIZER(Base, Base_);
    using Opt = typename Base::Optimizer;
    using Float = typename Base::Float;
    using Base::optimizer;

    HyperbolicPenalty(const Opt& optimizer, Float lambda0 = 1e1, Float tau0 = 1e1,
                      Float r = 1e1, Float q = 1e-1, const Stop& stop = Stop(), const Output& output = Output()) :
                      Base(optimizer, stop, output), lambda0(lambda0), tau0(tau0), r(r), q(q)
                      {
                      }

    HyperbolicPenalty(Float lambda0 = 1e1, Float tau0 = 1e1, Float r = 1e1, Float q = 1e-1,
                      const Stop& stop = Stop(), const Output& output = Output()) :
                      Base(stop, output), lambda0(lambda0), tau0(tau0), r(r), q(q)
                      {
                      }

    template <class Functions, class Domain, class Constraints>
    auto optimize (const Functions& funcs, const Domain& domain, const Constraints& constraints) const
    {
        return optimize(funcs, domain.x0, [&constraints](const auto& x){ return constraints.ineqs(x); }, stop, output);
    }

    template <class Function, class V, class Inequalities>
    V optimize (const Function&, V, const Inequalities&, Stop, Output) const;


    Float lambda0;
    Float tau0;

    Float r;
    Float q;
};

} // namespace impl

template <class Impl>
struct HyperbolicPenaltyBase : public Optimizer<Impl>
{
    NLPP_USING_OPTIMIZER(Base, Optimizer<Impl>);

    using Opt = typename traits::Optimizer<Impl>::Opt;
    using Float = typename traits::Optimizer<Impl>::Float;

    HyperbolicPenaltyBase(const Opt& optimizer = Opt(), const Stop& stop = Stop(), const Output& output = Output()) :
                          optimizer(optimizer), Base(stop, output) 
    {
    }

    Opt optimizer;
};

template <class Opt = LBFGS<>, class Stop = stop::GradientOptimizer<>, class Output = out::GradientOptimizer<>, typename Float = types::Float>
struct HyperbolicPenalty : public impl::HyperbolicPenalty<HyperbolicPenaltyBase<HyperbolicPenalty<Opt, Stop, Output, Float>>>
{
};

namespace traits
{

using ::nlpp::wrap::Conditions;

template <class Opt_, class Stop_, class Output_, class Float_>
struct Optimizer<HyperbolicPenalty<Opt_, Stop_, Output_, Float_>>
{
    using Opt = Opt_;
    using Stop = Stop_;
    using Output = Output_;
    using Float = Float_;

    static constexpr Conditions conditions = Conditions::Function | Conditions::Gradient |
                                             Conditions::Start    |
                                             Conditions::NLInequalities;
};

} // namespace traits

} // namespace nlpp
