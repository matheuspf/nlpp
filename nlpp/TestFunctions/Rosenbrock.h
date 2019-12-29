#pragma once

#include "../helpers/helpers.hpp"


namespace nlpp
{

struct Rosenbrock
{
	template <class Derived>
	double function (const Eigen::MatrixBase<Derived>& x) const
	{
		return operator()(x);
	}

	template <class Derived>
	double operator () (const Eigen::MatrixBase<Derived>& x) const
	{
		double r = 0.0;

        for(int i = 0; i < x.rows() - 1; ++i)
        	r += 100.0 * std::pow(x(i+1) - std::pow(x(i), 2), 2) + std::pow(x(i) - 1.0, 2);

        return r;
	}
};

} // namespace nlpp