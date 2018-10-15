#pragma once

#include "Helpers.h"



namespace nlpp
{

namespace out
{

template <typename Float>
struct GradientOptimizer<0, Float>
{
    void init (...) 
    {
    }

    void operator() (...)
    {
    }

    void finish (...)
    {
    }
};


template <typename Float>
struct GradientOptimizer<1, Float>
{
    template <class LineSearch, class Stop, class Output, class V>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx) 
    {
        handy::print("Init\n\n", "x:", x.transpose(), "\nfx:", fx, "\ngx:", gx.transpose()) << std::flush;
    }

    template <class LineSearch, class Stop, class Output, class V>
    void operator() (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                     const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx)
    {
        handy::print("x:", x.transpose(), "\nfx:", fx, "\ngx:", gx.transpose()) << std::flush;
    }

    template <class LineSearch, class Stop, class Output, class V>
    void finish (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx)
    {
        handy::print("x:", x.transpose(), "\nfx:", fx, "\ngx:", gx.transpose(), "\n\nFinish") << std::flush;
    }
};


template <class Float>
struct GradientOptimizer<2, Float>
{
    template <class LineSearch, class Stop, class Output, class V>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<V>& x, Float fx, const Eigen::MatrixBase<V>& gx) 
    {
        vX.clear();
        vFx.clear();
        vGx.clear();

        pushBack(x, fx, gx);
    }

    template <class LineSearch, class Stop, class Output, class V>
    void operator() (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                     const Eigen::MatrixBase<V>& x, Float fx, const Eigen::MatrixBase<V>& gx)
    {
        pushBack(x, fx, gx);
    }

    template <class LineSearch, class Stop, class Output, class V>
    void finish (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                 const Eigen::MatrixBase<V>& x, Float fx, const Eigen::MatrixBase<V>& gx)
    {
        pushBack(x, fx, gx);
    }


    template <class V>
    void pushBack (const Eigen::MatrixBase<V>& x, Float fx, const Eigen::MatrixBase<V>& gx)
    {
        vX.push_back(x);
        vFx.push_back(fx);
        vGx.push_back(gx);
    }


    std::vector<VecX<Float>> vX;
    std::vector<Float> vFx;
    std::vector<VecX<Float>> vGx;
};



} // namespace out

} // namespace nlpp