#pragma once

#include "direct_dec.hpp"


namespace nlpp::impl
{

template <class Base_>
template <class Func, class V>
V Direct<Base_>::optimize (const Func& func, const V& lower, const V& upper)
{
    V scale = upper - lower;

    auto scaledX = [&lower, &scale](const auto& x){
        return (lower.array() + x.array() * scale.array()).matrix();
    };

    auto scaledF = [&func, &scaledX](const auto& x){
        return func(scaledX(x));
    };

    int N = lower.size();

    Interval best = { func(V::Constant(N, 0.5)), 0.5*std::sqrt(N), std::vector<int>(N), V(V::Constant(N, 0.5)) };

    IntervalMap intervals;
    intervals[best.size].push(best);

    for(int iter = 0; iter < stop.maxIterations(); ++iter)
    {
        auto potSet = potentialSet(intervals, best);
        auto bestIter = createSplits(scaledF, potSet, intervals);

        best = std::min(best, bestIter);
    }

    return scaledX(best.x);
}


template <class Base_>
std::vector<typename Direct<Base_>::Interval> Direct<Base_>::potentialSet (IntervalMap& intervals, const Interval& best) const
{
    auto hull = convexHull(intervals);

    std::vector<Interval> potSet;
    potSet.reserve(hull.size());

    for(int i = 0; i < hull.size(); ++i)
    {
        double k1 = i > 0 ? (hull[i].fx - hull[i-1].fx) / (hull[i].size - hull[i-1].size) : -1e8;
        double k2 = i < hull.size() - 1 ? (hull[i].fx - hull[i+1].fx) / (hull[i].size - hull[i+1].size) : -1e8;
        double k = std::max(0.0, std::max(k1, k2));

        if(hull[i].fx - k * hull[i].size <= best.fx - eps * std::abs(best.fx) || i == hull.size() - 1)
            potSet.push_back(hull[i]);
    }

    return potSet;
}

template <class Base_>
template <class Func>
typename Direct<Base_>::Interval Direct<Base_>::createSplits (const Func& func, const std::vector<Interval>& potSet, IntervalMap& intervals) const
{
    Interval best{1e8};

    for(auto& interval : potSet)
    {
        int smallestK = *std::min_element(interval.k.begin(), interval.k.end());
        std::vector<std::tuple<Interval, Interval, int>> newIntervals;

        for(int i = 0; i < interval.k.size(); ++i) if(interval.k[i] == smallestK)
        {
            Interval left = interval;
            Interval right = interval;

            left.x(i) -= pow(3, -(interval.k[i] + 1));
            right.x(i) += pow(3, -(interval.k[i] + 1));

            left.fx = func(left.x);
            right.fx = func(right.x);

            newIntervals.emplace_back(std::move(left), std::move(right), i);
        }

        std::sort(newIntervals.begin(), newIntervals.end(), [](auto& ta, auto& tb){
            return std::min(std::get<0>(ta).fx, std::get<1>(ta).fx) < std::min(std::get<0>(tb).fx, std::get<1>(tb).fx);
        });

        std::vector<int> prevDims;

        for(auto& [left, right, k] : newIntervals)
        {
            prevDims.push_back(k);

            for(int prev : prevDims)
            {
                left.k[prev]++;
                right.k[prev]++;
            }

            left.size = intervalSize(left);
            right.size = intervalSize(right);

            best = std::min(best, std::min(left, right));

            intervals[left.size].push(std::move(left));
            intervals[right.size].push(std::move(right));
        }
    }

    return best;
}


template <class Base_>
std::vector<typename Direct<Base_>::Interval> Direct<Base_>::convexHull (IntervalMap& intervals) const
{
    std::vector<typename IntervalMap::iterator> hull;
    hull.reserve(intervals.size());

    const auto& first = intervals.begin()->second.top();
    const auto& last = std::prev(intervals.end())->second.top();

    double minSlope = (last.fx - first.fx) / std::max(last.size - first.size, 1e-8);

    for(auto it = intervals.begin(); it != intervals.end(); ++it)
    {
        if(it->second.top().fx > first.fx + (it->second.top().size - first.size) * minSlope)
            continue;

        while(hull.size() >= 2 && crossProduct(hull[hull.size()-2]->second.top(), hull[hull.size()-1]->second.top(), it->second.top()) <= 0)
            hull.pop_back();
        
        hull.push_back(it);
    }

    std::vector<Interval> hullValues;
    hullValues.reserve(hull.size());

    for(auto it : hull)
    {
        hullValues.emplace_back(it->second.top());
        it->second.pop();

        if(it->second.empty())
            it = intervals.erase(it);
        
        else
            it++;
    }

    return hullValues;
}

template <class Base_>
typename Direct<Base_>::Float Direct<Base_>::crossProduct (const Interval& o, const Interval& a, const Interval& b) const
{
    return (a.size - o.size) * (b.fx - o.fx) - (a.fx - o.fx) * (b.size - o.size);
}

template <class Base_>
float Direct<Base_>::intervalSize (const Interval& interval) const
{
    return 0.5 * std::sqrt(std::accumulate(interval.k.begin(), interval.k.end(), 0.0f, [](float sum, int k){
        return sum + std::pow(3, -2*k);
    }));
}

} // namespace nlpp::impl

