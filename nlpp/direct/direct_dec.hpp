#pragma once

#include <map>
#include <queue>

#include "../helpers/helpers.hpp"
#include "../utils/optimizer.hpp"


namespace nlpp
{

namespace impl
{

template <class Base_>
struct Direct : public Base_
{
// public:

    NLPP_USING_OPTIMIZER(Base, Base_);
    using Float = typename Base::Float;

    template <class Function, class V>
    V optimize (const Function&, const V&, const V&);

// protected:

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

    std::vector<Interval> potentialSet (IntervalMap& intervals, const Interval& best) const;
    template <class Func>
    Interval createSplits(const Func& func, const std::vector<Interval>& potSet, IntervalMap& intervals) const;
    std::vector<Interval> convexHull (IntervalMap& intervals) const;
    Float crossProduct (const Interval& o, const Interval& a, const Interval& b) const;
    float intervalSize (const Interval& interval) const;

    Float eps = 1e-4;
};

} // namespace impl


template <class Impl>
struct DirectBase : public BoundConstrainedOptimizer<Impl>
{
    NLPP_USING_OPTIMIZER(Base, BoundConstrainedOptimizer<Impl>);
    using Float = typename traits::Optimizer<Impl>::Float;
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