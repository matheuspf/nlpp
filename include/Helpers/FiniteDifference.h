#pragma once

#include "Helpers.h"


namespace cppnlp
{

struct Forward
{
    Forward () : fx(0.0) {}

    template <class F, typename Type>
    Forward (F f, const Type& x) : fx(f(x)) {}

    template <class F, typename Type>
    double operator () (F f, const Type& x, const Type& e, double h) const
    {
        return (f(x + e) - fx) / h;
    }

    double fx;
};

struct Backward
{
    Backward () : fx(0.0) {}

    template <class F, typename Type>
    Backward (F f, const Type& x) : fx(f(x)) {}

    template <class F, typename Type>
    double operator () (F f, const Type& x, const Type& e, double h) const
    {
        return (fx - f(x - e)) / h;
    }

    double fx;
};

struct Central
{
    Central (...) {}

    template <class F, typename Type>
    double operator () (F f, const Type& x, const Type& e, double h) const
    {
        return (f(x + e) - f(x - e)) / (2 * h);
    }
};




template <class F, class FType = Central, decltype(std::declval<F>()(double{}), int{}) = 0>
auto gradientFD (F f, FType = FType{}, double h = 1e-8)
{
    return [=](double x) -> double
    {
        return FType{f, x}(f, x, h, h);
    };
}



template <class F, class FType = Central, decltype(std::declval<F>()(Vec{}), int{}) = 0>
auto gradientFD (F f, FType = FType{}, double h = 1e-8)
{
    return [=](const Vec& x) -> Vec
    {
        Vec g(x.rows());
        Vec e = Vec::Zero(x.rows());

        FType fType{f, x};

        for(int i = 0; i < x.rows(); ++i)
        {
            e(i) = h;

            g(i) = fType(f, x, e, h);

            e(i) = 0.0;
        }

        return g;
    };
}



template <class F, decltype(std::declval<F>()(Vec{}), int{}) = 0>
auto hessianFD (F f, double h = 1e-8)
{
	return [=](Vec x) -> Mat
	{
		Mat hess(x.rows(), x.rows());

		Vec ei = Vec::Zero(x.rows());
		Vec ej = Vec::Zero(x.rows());

		for(int i = 0; i < x.rows(); ++i)
		{
			ei(i) = h;

			for(int j = i; j < x.rows(); ++j)
			{
				ej(j) = h;

				hess(i, j) = hess(j, i) = (f(x + ei + ej) - f(x + ei - ej) - 
										   f(x - ei + ej) + f(x - ei - ej)) / 
										   (4 * h * h);
				ej(j) = 0;
			}

			ei(i) = 0;
		}

		return hess;
	};
}


template <class F, decltype(std::declval<F>()(double{}), int{}) = 0>
auto hessianFD (F f, double h = 1e-8)
{
    return [=](double x) -> double
    {
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
    };
}

} // namespace cppnlp