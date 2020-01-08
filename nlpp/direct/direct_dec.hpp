#pragma once

#include <map>
#include <queue>

#include "utils/optimizer.hpp"
#include "utils/stop.hpp"
#include "utils/output.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
class Direct : public Base_
{
public:

    NLPP_USING_OPTIMIZER(Base, Base_);
    using Float = typename Base::Float;
    using Interval = typename Base::Interval;
    using IntervalComp = typename Base::IntervalComp;
    using IntervalMap = typename Base::IntervalMap;
    using Base::eps;

    template <class F, class V>
    V optimize (const F&, const V&, const V&);

private:

    std::vector<Interval> potentialSet (IntervalMap& intervals, const Interval& best);
    template <class Func>
    Interval createSplits(const Func& func, const std::vector<Interval>& potSet, IntervalMap& intervals);
    std::vector<Interval> convexHull (IntervalMap& intervals);
    Float crossProduct (const Interval& o, const Interval& a, const Interval& b) const;
    float intervalSize (const Interval& interval) const;
};

} // namespace impl


template <class Impl>
struct DirectBase : public BoundConstrainedOptimizer<Impl>
{
    NLPP_USING_OPTIMIZER(Base, BoundConstrainedOptimizer<Impl>);
    using Float = typename traits::Optimizer<Impl>::Float;

    struct Interval
    {
        Float fx;
        Float size;
        std::vector<int> k;
        Vec x;
    };

    struct IntervalComp
    {
        bool operator() (float a, float b) const
        {
            return a < b - 1e-10;
        }
    };

    friend bool operator< (const Interval& a, const Interval& b)
    {
        return a.fx < b.fx;
    }

    friend bool operator> (const Interval& a, const Interval& b)
    {
        return a.fx > b.fx;
    }

    using IntervalMap = std::map<Float, std::priority_queue<Interval, std::vector<Interval>, std::greater<Interval>>, IntervalComp>;

    Float eps = 1e-4;
};


template <class Stop = stop::Optimizer<>, class Output = out::Optimizer<>, typename Float = types::Float>
struct Direct : public impl::Direct<DirectBase<Direct<Stop, Output, Float>>>
{
};


namespace traits
{

template <class Stop_, class Output_, class Float_>
struct Optimizer<::nlpp::Direct<Stop_, Output_, Float_>>
{
    using Stop = Stop_;
    using Output = Output_;
    using Float = Float_;
};



} // namespace traits

} // namespace nlpp