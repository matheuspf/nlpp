#pragma once

#include "Helpers.h"


namespace nlpp
{

namespace stop
{

namespace impl
{

template <class Impl, typename Float>
struct GradientOptimizer
{
    GradientOptimizer(int maxIterations = 1000, double xTol = 1e-4, double fTol = 1e-4, double gTol = 1e-4) : 
                      maxIterations(maxIterations), xTol(xTol), fTol(fTol), gTol(gTol) {}

    template <class LineSearch, class Stop, class Output, class V>
    void init (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx) 
    {
        fx0 = fx;
        x0 = x;
        gx0 = gx;
    }


    template <class LineSearch, class Stop, class Output, class V>
    bool operator () (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
                      const Eigen::MatrixBase<V>& x, double fx, const Eigen::MatrixBase<V>& gx) 
    {
        bool xStop = (x - x0).norm() < xTol;
        bool fStop = std::abs(fx - fx0) < fTol;
        bool gStop = gx.norm() < gTol;

        fx0 = fx;
        x0 = x;
        gx0 = gx;

        return static_cast<Impl&>(*this).stop(xStop, fStop, gStop);
    }



    Float fx0;

    VecX<Float> x0;

    VecX<Float> gx0;


    int maxIterations;      ///< Maximum number of outer iterations

	Float xTol;            ///< Minimum tolerance on the norm of the input (@c x) between iterations

	Float fTol;            ///< Minimum tolerance on the value of the function (@c x) between iterations

    Float gTol;            ///< Minimum tolerance on the norm of the gradient (@c g) between iterations

};


} // namespace impl



template <bool Exclusive, typename Float>
struct GradientOptimizer : public impl::GradientOptimizer<GradientOptimizer<Exclusive, Float>, Float>
{
    using impl::GradientOptimizer<GradientOptimizer<Exclusive, Float>, Float>::GradientOptimizer;

    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop && fStop && gStop;
    }
};


template <typename Float>
struct GradientOptimizer<false, Float> : public impl::GradientOptimizer<GradientOptimizer<false, Float>, Float>
{
    using impl::GradientOptimizer<GradientOptimizer<false, Float>, Float>::GradientOptimizer;

    bool stop (double xStop, double fStop, double gStop)
    {
        return xStop || fStop || gStop;
    }
};


} // namespace stop

} // namespace nlpp