#pragma once

#include "nlpp/direct/direct_dec.hpp"


namespace nlpp::poly
{

template <class V = nlpp::Vec>
struct DirectBase : public nlpp::poly::BoundConstrainedOptimizer<V>
{
    NLPP_USING_BOUND_CONSTRAINED_OPTIMIZER(Base, nlpp::poly::BoundConstrainedOptimizer<V>);
    using Float = nlpp::impl::Scalar<V>;
};

template <class V = ::nlpp::Vec>
struct Direct : public ::nlpp::impl::Direct<DirectBase<V>>
{
    NLPP_USING_BOUND_CONSTRAINED_OPTIMIZER(Base, ::nlpp::impl::Direct<DirectBase<V>>);

    virtual V optimize (const ::nlpp::wrap::poly::Function<V>&, const V&, const V&);
    virtual Direct<V>* clone () const;
};

} // namespace poly
