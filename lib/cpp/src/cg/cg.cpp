#include "nlpp/cg/cg.hpp"
#include "lib/cpp/include/cg/cg.hpp"

namespace nlpp_p
{

template <class V>
V CG<V>::optimize (nlpp::wrap::poly::FunctionGradient<V> f, V x)
{
	return Base::optimize(f, x);
}

template <class V>
CG<V>* CG<V>::clone () const
{
    return new CG(*this);
}

template class CGBase<>;
template class CG<>;

} // namespace nlpp_p