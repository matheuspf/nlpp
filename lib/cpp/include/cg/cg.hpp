#pragma once

#include "nlpp/cg/cg_dec.hpp"
#include "projections.hpp"

namespace nlpp::poly
{

template <class V = nlpp::Vec>
struct CGBase : public nlpp::poly::LineSearchOptimizer<V>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, nlpp::poly::LineSearchOptimizer<V>);

    using CGType = Projection<V>;

    CGType cg;
	double v = 0.1;
};

template <class V = nlpp::Vec>
struct CG : public nlpp::impl::CG<CGBase<V>>
{
    NLPP_USING_LINESEARCH_OPTIMIZER(Base, nlpp::impl::CG<CGBase<V>>);

	virtual V optimize (const nlpp::wrap::poly::FunctionGradient<V>&, const V&) const;
	virtual CG<V>* clone () const;
};


} // namespace nlpp::poly