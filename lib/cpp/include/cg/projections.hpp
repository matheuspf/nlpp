#pragma once

#include "nlpp/CG/projections_dec.hpp"

#define NLPP_CG_PROJECTION_DEC_POLY(NAME) \
template <class V = nlpp::Vec> \
struct NAME : public nlpp::NAME, public ProjectionBase<V>  \
{   \
    using Base = ProjectionBase<V>; \
    using Impl = nlpp::NAME;   \
    using Impl::Impl;          \
    virtual nlpp::impl::Scalar<V> operator () (const V& fa, const V& fb, const V& dir = V{}) const; \
    virtual NAME* clone_impl () const { return new NAME(*this); }   \
};

namespace nlpp_p
{

template <class V = nlpp::Vec>
struct ProjectionBase : public nlpp::poly::CloneBase<ProjectionBase<V>>
{
    virtual ~ProjectionBase () = 0;
    virtual nlpp::impl::Scalar<V> operator () (const V&, const V&, const V& = V{}) const = 0;
};

template <class V = nlpp::Vec>
struct Projection : public ::nlpp::poly::PolyClass<ProjectionBase<V>>
{
    NLPP_USING_POLY_CLASS(Projection, Base, ::nlpp::poly::PolyClass<ProjectionBase<V>>);

    Projection();
    nlpp::impl::Scalar<V> operator () (const V& fa, const V& fb, const V& dir = V{}) const;
};

NLPP_CG_PROJECTION_DEC_POLY(FR)
NLPP_CG_PROJECTION_DEC_POLY(PR)
NLPP_CG_PROJECTION_DEC_POLY(PR_Abs)
NLPP_CG_PROJECTION_DEC_POLY(PR_Plus)
NLPP_CG_PROJECTION_DEC_POLY(HS)
NLPP_CG_PROJECTION_DEC_POLY(DY)
NLPP_CG_PROJECTION_DEC_POLY(HZ)
NLPP_CG_PROJECTION_DEC_POLY(FR_PR)

} // namespace nlpp_p

#undef NLPP_CG_PROJECTION_DEC_POLY