#pragma once

#include "nlpp/CG/CG_dec.h"
#include "projections.hpp"

namespace nlpp_p
{

template <class V = nlpp::Vec>
struct CGBase : public nlpp::poly::LineSearchOptimizer<V>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base,  nlpp::poly::LineSearchOptimizer<V>);

    Projection<V> cg;
	double v = 0.1;
};

template <class V = nlpp::Vec>
struct CG : public nlpp::impl::CG<CGBase<V>>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, nlpp::impl::CG<CGBase<V>>);

	virtual V optimize (nlpp::wrap::poly::FunctionGradient<V> f, V x);
	virtual CG<V>* clone () const;
};


} // namespace nlpp_p