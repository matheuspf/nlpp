#include "nlpp/cg/cg.hpp"
#include "lib/cpp/include/cg/cg.hpp"

namespace nlpp::poly
{

template <class V>
V CG<V>::optimize (const ::nlpp::wrap::poly::FunctionGradient<V>& func, const V& x) const
{
	return Base::optimize(func, x);
}

template <class V>
CG<V>* CG<V>::clone () const
{
    return new CG(*this);
}

template class CGBase<>;
template class CG<>;

} // namespace nlp::poly