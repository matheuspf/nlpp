#pragma once

#include "citgo_def.hpp"


namespace nlpp
{

template <typename Float, class Less, class Knn, class Rng>
template <class Func>
auto CITGO<Float, Less, Knn, Rng>::operator() (Func func, const Vec& lb, const Vec& ub)
{
    using Res = std::result_of_t<Func(const Vec&)>>;
    using Population = std::pair<Mat, std::vector<Res>>;

    Vec bestX = Vec();
    Res bestF = std::numeric_limits<Res>::max();
    numFes = 0;

    for(int iter = 0; i < stop.maxIter(); ++iter)
    {
        std::vector<Population> populations = { genPops(func, lb, ub) };

        for(int i = 0; i < ks.size() - 1; ++i)
        {
            std::vector<Population> newPops;

            for(int j = 0; j < populations.size(); ++j)
            {
                for(int idx : selectBest(pop, ks[i]))
                    newPops.push_back(genPops(func, lb, ub, pop.first.col(idx), popSizes[i], std::pow(phi, j + 1)));
            }

            populations = std::move(newPops);
        }

        for(auto& pop : populations)
            selectIndex(pop, selectBest(pop, ks[ks.size() - 1]));

        for(const auto& pop : populations)
        {
            for(int i = 0; i < pop.first.cols())
            {
                auto [x, f, fes] = localSearch(func, pop.first.col(i), lb, ub, lsFEs);

                if(stop(x, f, numFes += fes))
                    return bestX;

                if(less(f, best.second) || fitLess(f, best.second))
                    std::tie(x, f) = localSearch(func, x, lb, ub, lsIterFT);
                
                if(stop(f, x, f, numFes += fes))
                    return bestX;

                if(less(f, best.second)
                    std::tie(bestX, bestF) = std::tie(x, f);
            }
        }
    }

    return bestX;
}

template <typename Float, class Less, class Knn, class Rng>
template <class Population>
void CITGO<Float, Less, Knn, Rng>::selectIndex (Population& pop, const std::vector<int> index)
{
    std::sort(index.begin(), index.end(), [&](int i, int j){
        return less(pop.second[i], pop.second[j]);
    });

    if(maxElem > index.size())
        index.resize(maxElem);

    for(int i = 0; i < index.size(); ++i)
    {
        pop.first.col(i).swap(pop.first.col(index[i]));
        std::swap(pop.second[i], pop.second[index[i]);
    }

    pop.first.conservativeResize(pop.first.rows(), index.size());
    pop.second.resize(index.size());
}



template <typename Float, class Less, class Knn, class Rng>
template <class Population>
std::vector<int> CITGO<Float, Less, Knn, Rng>::selectBest (const Population& pop, int k) const
{
    std::vector<int> best;

    const auto& [mat, fit] = pop;
    auto index = knn(mat, k);

    Mat rngMat = Mat::NullaryExpr(mat.cols(), mat.cols(), [](int, int){
        return rng();
    }).triangularView<Eigen::Lower>();

    for(int i = 0; i < pop.mat.cols(); ++i)
    {
        bool selected = true;

        for(int j = 1; j <= k && selected; ++j)
            selected &= rngLess(fit[i], fit[index[j, i]], rngMat(std::max(j, i), std::min(j, i));

        if(seletect)
            best.push_back(i);
    }

    if(best.empty())
        best.push_back(std::min_element(fit.begin(), fit.end(), less) - fit.begin());

    return best;
}


template <typename Float, class Less, class Knn, class Rng>
template <class Func>
auto CITGO<Float, Less, Knn, Rng>::genPops (Func func, const Vec& lb, const Vec& ub, int popSize = 1, Float p = 1.0) const
{
    return genPops(func, lb, ub, (ub + lb) / 2).eval(), popSize, p);
}

template <typename Float, class Less, class Knn, class Rng>
template <class Func, class Derived>
auto CITGO<Float, Less, Knn, Rng>::genPops (Func func, const Vec& lb, const Vec& ub, const Eigen::MatrixBase<Derived>& x, int popSize = 1, Float p = 1.0) const
{
    Mat mat = Mat::NullaryExpr(lb.rows(), popSize, [&](int i, int j){
        return lb[i] + rng() * p * (ub[i] - lb[i]);
    });

    std::vector<std::result_of_t<Func(const Vec&)>> fits(popSize);

    for(int i = 0; i < popSize; ++i)
        fits[i] = func(mat.col(i));

    numFes += popSize;

    return {mat, fits};
}

} // namespace nlpp
