#pragma once

#include "../Helpers/Helpers.h"

#include "../Helpers/FiniteDifference.h"


namespace cppnlp
{

template <class Direction>
struct TrustRegion
{
	template <class Function, class Gradient, class Hessian>
	Vec operator () (Function function, Gradient gradient, Hessian hessian, Vec x)
	{
		double delta = delta0;

		double fx = function(x);
		Vec gx = gradient(x);
		Mat hx = hessian(x);


		//db(x.transpose(), "       ", gx.transpose(), "\n\n", hx); exit(0);


		for(int iter = 0; iter < maxIter; ++iter)
		{
			/** Making a call to the actual function that generates the direction whitin the trust region.
			  *	I am using CRTP here, so the 'Direction' class inherits from this class. **/
			Vec dir = static_cast<Direction&>(*this).direction(function, gradient, hessian, x, delta, fx, gx, hx);

			Vec y = x + dir;			/// New point and
			double fy = function(y);	/// and its fitness

			double aRed = (fx - fy);	/// Actual reduction 
			double pRed = -(gx.dot(dir) + 0.5 * dir.transpose() * hx * dir);	/// Predicted reduction

			/** Ratio between actual and predicted reduction. 'pRed' SHOULD always be positive, but I am also 
			 *  handling the case where the trust region direction fails miserably and returns a worst solution.
			 *  That 'abs' actually should not be there.
			**/
			double rho = aRed / std::max(constants::eps, std::abs(pRed));

			//db(delta, "       ", function(x), "      ", aRed, "      ", pRed, "       ", rho, "       ", x.transpose(), "         ", dir.transpose(), "\n\n\n");
			//db((fx - fy), "      ", (-gx.dot(dir) - 0.5 * dir.transpose() * hx * dir), "       ", x.transpose(), "     ", dir.transpose(), "\n\n\n");

			if(std::isnan(rho))
			{
				handy::print("\n\n\n", fx, "     ", fy, "          ", gx.transpose(), "          ", dir, "\n\n");
				handy::print(hx); exit(0);
			}



			/// If this is the maximum allowed value for delta and we had this small improvement, theres nothing else to do
			if(aRed < constants::eps && delta == maxDelta)
				return x;

			/// If rho is smaller than this threshold, reduce the trust region size
			if(rho < alpha)
				delta = alpha * delta;

			/** If rho is greater than this other threshold AND the direction is at
			  *	the border of the trust region, then increase the trust region size.
			  *
			  * TODO: change that aproximation (1e-4) and return a flag telling if the direction is at the border.
			**/
			else if(rho > 1.0 - alpha && std::abs(dir.norm() - delta) < 1e-4)
				delta = std::min(beta * delta, maxDelta);


			/// Too small trust region, go home. Nothing else to do
			if(delta < 2 * constants::eps) 
				return x;

			/// We only update the actual x if rho is greater than eta and the function value is actually reduced
			if(rho > eta)
			{
				x = y;
				fx = fy;
				gx = gradient(x);
				hx = hessian(x);
			}
		}

		return x;
	}



	template <class Function, class Gradient>
	Vec operator () (Function f, Gradient g, const Vec& x)
	{
		return this->operator()(f, g, hessianFD(f), x);
	}

	template <class Function>
	Vec operator () (Function f, const Vec& x)
	{
		return this->operator()(f, gradientFD(f), x);
	}



	double delta0;
	double alpha;
	double beta;
	double eta;

	int maxIter;
	double maxDelta;


private:

	TrustRegion (double delta0 = 10.0, double alpha = 0.25, double beta = 2.0,
			 double eta = 0.1, int maxIter = 1e5, double maxDelta = 1e2) :
			 delta0(delta0), alpha(alpha), beta(beta), eta(eta),
			 maxIter(maxIter), maxDelta(maxDelta) {}


	friend Direction;
};

} // namespace cppnlp
