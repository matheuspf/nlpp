#include "nlpp/cg/cg.hpp"
#include "lib/cpp/include/cg/cg.hpp"

namespace nlpp::poly
{

template <class V>
V CG<V>::optimize (::nlpp::wrap::poly::FunctionGradient<V> func, V x, const LineSearch& lineSearch, const Stop& stop, const Output& output) const
{
	return Base::optimize(func, x, lineSearch, stop, output);
}

template <class V>
CG<V>* CG<V>::clone () const
{
    return new CG(*this);
}

template class CGBase<>;
template class CG<>;

} // namespace nlp::poly