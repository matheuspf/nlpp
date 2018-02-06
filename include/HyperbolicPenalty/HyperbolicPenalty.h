#ifndef OPT_HYPERBOLIC_PENALTY_H
#define OPT_HYPERBOLIC_PENALTY_H

#include "../Modelo.h"
#include "../FiniteDifference.h"
#include "../QuasiNewton/BFGS/BFGS.h"


template <class Optimizer = BFGS<StrongWolfe, BFGS_Constant>>
struct HyperbolicPenalty
{
    HyperbolicPenalty(const Optimizer& optimizer = Optimizer(), double lambda0 = 1e1, double tau0 = 1e1, 
                      double r = 1e1, double q = 1e-1, int maxIter = 10, double gTol = 1e-6) :
                      optimizer(optimizer), lambda0(lambda0), tau0(tau0), r(r), q(q), maxIter(maxIter), gTol(gTol) {}

    HyperbolicPenalty (double lambda0, double tau0 = 1e1, double r = 1e1, double q = 1e-1, 
                       int maxIter = 10, double gTol = 1e-6, const Optimizer& optimizer = Optimizer()) :
                       optimizer(optimizer), lambda0(lambda0), tau0(tau0), r(r), q(q), maxIter(maxIter), gTol(gTol) {}



    template <class Function, class Inequalities>
    Vec operator () (Function&& function, Inequalities&& inequalities, Vec x)
    {
        double lambda = lambda0, tau = tau0;

        auto penalFunc = [&](const Vec& x) -> double
        {
            const ArrayXd& ineq = -inequalities(x).array();

            return function(x) + (-lambda * ineq + sqrt(pow(lambda, 2) * pow(ineq, 2) + pow(tau, 2))).sum();
        };

        auto penalGrad = gradientFD(penalFunc);
        
        
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




#endif // OPT_HYPERBOLIC_PENALTY_H