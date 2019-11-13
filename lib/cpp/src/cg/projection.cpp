#include "nlpp/CG/projections.hpp"
#include "lib/cpp/include/cg/projections.hpp"

#define NLPP_CG_PROJECTION_POLY(NAME) \
template <class V> \
nlpp::impl::Scalar<V> NAME<V>::operator () (const V& fa, const V& fb, const V& dir) const \
{   \
    return Impl::operator()(fa, fb, dir);   \
}   \
template class NAME<>;

namespace nlpp_p
{

template <class V>
Projection<V>::Projection() : Base(std::make_unique<FR_PR<V>>())
{
}

template <class V>
nlpp::impl::Scalar<V> Projection<V>::operator () (const V& fa, const V& fb, const V& dir) const
{
    return impl->operator()(fa, fb, dir);
}

template class ProjectionBase<>;
template class Projection<>;

NLPP_CG_PROJECTION_POLY(FR)
NLPP_CG_PROJECTION_POLY(PR)
NLPP_CG_PROJECTION_POLY(PR_Abs)
NLPP_CG_PROJECTION_POLY(PR_Plus)
NLPP_CG_PROJECTION_POLY(HS)
NLPP_CG_PROJECTION_POLY(DY)
NLPP_CG_PROJECTION_POLY(HZ)
NLPP_CG_PROJECTION_POLY(FR_PR)


} // namespace nlpp_p

#undef NLPP_CG_PROJECTION_POLY