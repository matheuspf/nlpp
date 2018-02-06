#pragma once

#include "../Helpers/Helpers.h"

#include "../LineSearch/StrongWolfe/StrongWolfe.h"


#define BUILD_CG_STRUCT(Op, ...) \
\
struct __VA_ARGS__ \
{\
	double operator () (const Vec& fa, const Vec& fb, const Vec& dir = Vec()) const \
	{\
		Op; \
	}\
};


BUILD_CG_STRUCT(return fb.dot(fb) / fa.dot(fa), FR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / fa.dot(fa), PR);

BUILD_CG_STRUCT(return abs(PR::operator()(fa, fb)), PR_Abs : PR);

BUILD_CG_STRUCT(return max(0.0, PR::operator()(fa, fb)), PR_Plus : PR);

BUILD_CG_STRUCT(return fb.dot(fb - fa) / dir.dot(fb - fa), HS);

BUILD_CG_STRUCT(return fb.dot(fb) / dir.dot(fb - fa), DY);

BUILD_CG_STRUCT(Vec y = fb - fa;

				double yp = y.dot(dir);
				double yy = y.dot(y);

				return (y - 2 * dir * (yy / yp)).dot(fb) / yp;,

				HZ);

BUILD_CG_STRUCT(double fr = FR::operator()(fa, fb);
				double pr = PR::operator()(fa, fb);

				if(pr < -fr) return -fr;

				if(abs(pr) <= fr) return pr;

				return fr;,

				PR_FR : FR, PR);







template <class CGType = PR_FR, class LineSearch = StrongWolfe>
struct CG
{
	CG(LineSearch lineSearch = StrongWolfe(1e-2, 1e-4, 0.1)) : lineSearch(lineSearch) {}


	template <class Function, class Gradient>
	Vec operator () (Function function, Gradient gradient, Vec x)
	{
		Vec fa = gradient(x), fb, dir = -fa;

		for(int iter = 0; iter < maxIterations * x.rows() && norm(dir) > xTol; ++iter)
		{
			double alpha = lineSearch(function, gradient, x, dir);

			x = x + alpha * dir;

			fb = gradient(x);


			if((fa.dot(fb) / fb.dot(fb)) >= v)
				dir = -fb;

			else
				dir = -fb + cg(fa, fb, dir) * dir;
		

			fa = fb;
		}

		return x;
	}


	template <class Function>
	Vec operator () (Function function, Vec x)
	{
		return this->operator()(function, gradientFD(function), x);
	}




	CGType cg;

	LineSearch lineSearch;


	double v = 0.1;

	int maxIterations = 1e3;

	double xTol = 1e-6;

	double fTol = 1e-7;

};













#endif // endif OPT_CG_H