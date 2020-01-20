#include "nlpp/gradient_descent/gradient_descent.hpp"
#include "lib/cpp/include/gradient_descent/gradient_descent.hpp"

namespace nlpp::poly
{

template <class V>
V GradientDescent<V>::optimize (const ::nlpp::wrap::poly::FunctionGradient<V>& func, const V& x)
{
	return Base::optimize(func, x);
}

template <class V>
GradientDescent<V>* GradientDescent<V>::clone () const
{
    return new GradientDescent(*this);
}

template class GradientDescent<>;

} // namespace nlp::poly