#ifndef OPT_STRONG_WOLFE_LS
#define OPT_STRONG_WOLFE_LS 

#include <type_traits>

#include <Eigen/Dense>

namespace opt
{
	using Eigen::VectorXd;

	struct StrongWolfeLS
	{
		template <class Function, class Gradient>
		double operator () (Function f, Gradient g, const VectorXd& x, const VectorXd& d)
		{
			double ap = 0.0, ai = 1e-4;

			double f0 = f(x);
			double g0 = g(x).transpose() * d;

			double fp = f0;
			double fMax = f(x + aMax * d);


			for(int i = 1; i <= maxIter; ++i)
			{
				double fi = f(x + ai * d);

				if(fi > f0 + c1 * ai * g0 || (i > 1 && fi >= fp))
					return zoom(ap, ai, x, d, f, g, f0, g0);

				double gi = g(x + ai * d).transpose() * d;

				if(std::abs(gi) <= -c2 * g0)
					return ai;

				if(gi >= 0)
					return zoom(ai, ap, x, d, f, g, f0, g0);

				ap = ai;
				fp = fi;

				ai = interpolate(ai, fi, gi, aMax, fMax);
			}

			return ai;
		}


		inline double interpolate (double a, double fa, double ga, double b, double fb)
		{
			return - (ga * b * b) / (2.0 * (fb - fa - ga * b));
		}


		// inline double interpolate (double a, double fa, double ga, double b, double fb)
		// {
		// 	return (a + b) / 2.0;
		// }



		template <class F, class G>
		double zoom (double alo, double ahi, const Eigen::VectorXd& x, const Eigen::VectorXd& d,
					 F f, G g, 
					 double f0, double g0)
		{
			double aj;

			double flo = f(x + alo * d);
			double fhi = f(x + ahi * d);
			double glo = g(x + alo * d).transpose() * d;


			while(maxIter--)
			{
				//DB(alo << "   " << flo << "   " << glo << "   " << ahi << "   " << fhi);

				aj = interpolate(alo, flo, glo, ahi, fhi);

				//DB(aj);

				double fj = f(x + aj * d);

				if(fj > f0 + c1 * aj * g0 || fj >= flo)
				{
					ahi = aj;
					fhi = fj;
				}

				else
				{
					double gj = g(x + aj * d).transpose() * d;

					if(std::abs(gj) <= -c2 * g0)
						return aj;

					if(gj * (ahi - alo) >= 0)
						ahi = alo;

					alo = aj;
					flo = fj;
					glo = gj;
				}
			}

			return aj;
		}


			int maxIter = 10;

			double c1 = 1e-4;
			double c2 = 0.9;

			double a0 = 1.0;
			double aMax = 5.0;
		};


} // namespace opt




#endif // OPT_STRONG_WOLFE_LS