#pragma once

#include "../helpers/helpers.hpp"

#include "../Helpers/FiniteDifference.h"

#include "../QuasiNewton/LBFGS/LBFGS.h"


namespace nlpp
{

template <class Optimizer = LBFGS<>>
struct HyperbolicPenalty
{
    HyperbolicPenalty(const Optimizer& optimizer = Optimizer(), double lambda0 = 1e1, double tau0 = 1e1, 
                      double r = 1e1, double q = 1e-1, int maxIter = 10, double gTol = 1e-6) :
                      optimizer(optimizer), lambda0(lambda0), tau0(tau0), r(r), q(q), maxIter(maxIter), gTol(gTol) {}

    HyperbolicPenalty (double lambda0, double tau0 = 1e1, double r = 1e1, double q = 1e-1, 
                       int maxIter = 10, double gTol = 1e-6, const Optimizer& optimizer = Optimizer()) :
                       optimizer(optimizer), lambda0(lambda0), tau0(tau0), r(r), q(q), maxIter(maxIter), gTol(gTol) {}



    template <class Function, class Inequalities>
    Vec operator () (Function function, Inequalities inequalities, Vec x)
    {
        double lambda = lambda0, tau = tau0;

        auto penalFunc = [&](const Vec& x) -> double
        {
            auto ineq = -inequalities(x).array();

            return function(x) + (-lambda * ineq + Eigen::sqrt(std::pow(lambda, 2) * Eigen::pow(ineq, 2) + std::pow(tau, 2))).sum();
        };

        auto penalGrad = fd::gradient(penalFunc);
        
        
        for(int iter = 0; iter < maxIter && penalGrad(x).norm() > gTol; ++iter)
        {
            x = optimizer(penalFunc, penalGrad, x);

            if((inequalities(x).array() > 0.0).any())
                lambda = r * lambda;

            else
                tau = q * tau;
        }

        return x;
    }


    Optimizer optimizer;

    double lambda0;
    double tau0;

    double r;
    double q;

    int maxIter;

    double gTol;
};

} // namespace nlpp
