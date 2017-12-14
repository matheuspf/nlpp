#ifndef OPT_LINE_SEARCH_H
#define OPT_LINE_SEARCH_H

#include "../Modelo.h"

#include "../FiniteDifference.h"


template <class Impl>
class LineSearch
{
public:

	template <class Function, class Gradient>
	double operator () (Function f, Gradient g)
	{
		return static_cast<Impl&>(*this).lineSearch(f, g);
	}

	template <class Function, class Gradient>
	double operator () (Function f, Gradient g, double x, double dir = 1.0)
	{
		N = 1;
		xNorm = std::abs(x);

		return this->operator()([&](double a){ return f(x + a * dir); },
								[&](double a){ return g(x + a * dir) * dir; });
	}

	template <class Function, class Gradient>
	double operator () (Function f, Gradient g, const Vec& x, const Vec& dir)
	{
		N = x.rows();
		xNorm = x.norm();

		return this->operator()([&](double a){ return f(x + a * dir); },
								[&](double a){ return g(x + a * dir).dot(dir); });
	}


	template <class Function>
	double operator () (Function f, double x, double dir = 1.0)
	{
		return this->operator()(f, gradientFD(f), x, dir);
	}

	template <class Function>
	double operator () (Function f, const Vec& x, const Vec& dir)
	{
		return this->operator()(f, gradientFD(f), x, dir);
	}


private:

	LineSearch () {}

	friend Impl;

	int N;

	double xNorm;
};




#endif // OPT_LINE_SEARCH_H