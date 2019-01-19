#include "Helpers.h"

#define NLPP_CHECK_IF_OVERRIDEN(Method, Obj) ((void*)(Obj->*(&Method)) != (void*)(&Method))



namespace nlpp
{

namespace poly
{

template <class V = Vec>
struct Function
{
    using V = V_;
    using Float = impl::Scalar<V>;

    virtual Float function (const Eigen::Ref<const V>&)
    {
        return Float{};
    }
    
    virtual V gradient (const Eigen::Ref<const V>&)
    {
        return V{};
    }

    virtual void gradient (const Eigen::Ref<const V>&, Eigen::Ref<V>)
    {
    }

    virtual std::pair<Float, V> functionGradient (const Eigen::Ref<const V>&)
    {
        return std::make_pair(function(V), gradient(V));
    }

    virtual Float functionGradient (const Eigen::Ref<const V>&, Eigen::Ref<V>)
    {
        return function(V);
    }

    virtual Function* clone ()
    {
        return new Function{};
    }
};

} // namespace poly



} // namespace nlpp