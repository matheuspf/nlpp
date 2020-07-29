#pragma once

#include "helpers/helpers.hpp"
#include "knn/kdtree_def.hpp"


namespace nlpp
{

template <class Comp = std::less<void>, typename Float = types::Float>
struct EpsComp : public Comp
{
    using Comp::operator();

    template <typename T>
    bool operator () (const std::pair<T, T>& x, const std::pair<T, T>& y) const
    {
        return std::abs(x.first - y.first) < eps ? Comp::operator()(x.second, y.second) : Comp::operator()(x, y);
    }

    Float eps = Float{1e-4};
};

template <class Comp = std::less<void>>
struct RngComp : public Comp
{
    using Comp::operator();

    template <typename T>
    bool operator () (const std::pair<T, T>& x, const std::pair<T, T>& y, double rng = 1.0) const
    {
        return rng < 0.5 ? Comp::operator()(x.second, y.second) : Comp::operator()(x, y);
    }
};

template <class Comp = std::less<void>>
struct FitnessOnlyComp : public Comp
{
    using Comp::operator();

    template <typename T>
    bool operator () (const std::pair<T, T>& x, const std::pair<T, T>& y) const
    {
        return Comp::operator()(x.second, y.second);
    }
};


template <typename Float = types::Float, class Less = EpsComp<Float>, class Knn = KDTreeKNN, class Rng = handy::Rand<Float>>
struct CITGO
{
    CITGO(const std::vector<int>& popSizes, int K, Float prob, Float phi, int maxFEs = 1e3)
    
    using Mat = MatX<Float>;
    using Vec = VecX<Float>;
    using RngLess = RngComp<Less>;
    using FitLess = FitnessOnlyComp<Less>;


    template <class Func>
    void initialize (Func func, int iter);

    template <class Func>
    auto genPops (Func func, const Vec& lb, const Vec& ub, int popSize = 1, Float p = 1.0) const;

    template <class Func, class Derived>
    auto genPops (Func func, const Vec& lb, const Vec& ub, const Eigen::MatrixBase<Derived>& x, int popSize = 1, Float p = 1.0) const;



    template <class Population>
    std::vector<int> selectBest (const Population& pop, int k) const;

    template <class Population>
    void selectIndex (Population& pop, const std::vector<int> index);

    template <class Func>
    auto operator() (Func func, const Vec& lb, const Vec& ub);


    int maxIter;
    std::vector<int> ks;

    Float phi;
    Float prob;
    Float mu;
    Float Eps;

    std::vector<int> popSizes;

    int numFes;
    int maxElem;
    int lsFEs;

    // std::pair<Vec, std::pair<Float, Float>> best;

    Less less;
    RngLess rngLess;
    FitLess fitLess;

    Knn knn;
    Rng rng;
};

} // namespace nlpp
