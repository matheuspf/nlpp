#pragma once

#include "nlpp/gradient_descent/gradient_descent_dec.hpp"

namespace nlpp::poly
{

template <class V = nlpp::Vec>
struct GradientDescent : public nlpp::impl::GradientDescent<nlpp::poly::LineSearchOptimizer<V>>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, nlpp::impl::GradientDescent<nlpp::poly::LineSearchOptimizer<V>>);

	virtual V optimize (const nlpp::wrap::poly::FunctionGradient<V>&, V);
	virtual GradientDescent<V>* clone () const;
};


} // namespace nlpp::poly