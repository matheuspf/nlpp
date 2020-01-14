#include "lib/cpp/include/direct/direct.hpp"
#include "nlpp/direct/direct.hpp"

namespace nlpp::poly
{

template <class V>
V Direct<V>::optimize (const ::nlpp::wrap::poly::Function<V>& func, V lower, V upper)
{
	return Base::optimize(func, lower, upper);
}

template <class V>
Direct<V>* Direct<V>::clone () const
{
    return new Direct(*this);
}

template class DirectBase<>;
template class Direct<>;

};