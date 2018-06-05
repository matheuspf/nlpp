#pragma once

#include "../TrustRegion.h"

#include "../../Helpers/SpectraHelpers.h"


namespace nlpp
{

struct IterativeTR : public TrustRegion<IterativeTR>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, 
				   double delta, double fx, const Vec& gx, Mat hx)
	{
        int N = x.rows();

        Vec p, q;
        
        Mat In = Mat::Identity(N, N);
        

        if(tryFactorize(gx, hx, p, delta))
            return p;



        TopEigen<Spectra::SELECT_EIGENVALUE::SMALLEST_ALGE> topEigen(hx, 2);
        
        double lambda = 2 * std::abs(topEigen.eigenvalues()(0));

        Vec v1 = topEigen.eigenvectors().col(0);


        if(std::abs(gx.dot(v1)) < 1e-4)
        {
            Eigen::LLT<Mat> llt(hx + std::abs(topEigen.eigenvalues()(0)) * In);

            p = -llt.solve(gx);

            return p + v1 * ((delta - p.norm()) / v1.norm());
        }
        

        int maxIters = 3;

        while(maxIters--)
        {
            Eigen::LLT<Mat> llt(hx + lambda * In);

            // if(llt.info() == Eigen::NumericalIssue)
            // {
            //     maxIters = 3;
            //     lambda = 4 * std::abs(topEigen.eigenvalues()(0));
            //     continue;
            // }

            // db(abs(p.norm() - delta), "     ", (abs(p.norm() - delta) <= 1e-4));


            p = -llt.solve(gx);

            if(std::abs(p.norm() - delta) <= 1e-4)
                return p;


            Mat L = llt.matrixL();

            q = L.triangularView<Eigen::Lower>().solve(p);
            
            
            lambda = lambda + (p.squaredNorm() / q.squaredNorm()) * ((p.norm() - delta) / delta);
        }

        return p;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
    }


    bool tryFactorize (const Vec& gx, const Mat& hx, Vec& p, double delta)
    {
        Eigen::LLT<Mat> llt(hx);
        
        p = -llt.solve(gx);

        if(p.norm() <= delta)
            return true;

        return false;
    }
};

} // namespace nlpp
