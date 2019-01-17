#include "Helpers.h"

#define NLPP_CHECK_IF_OVERRIDEN(Method, Obj) ((void*)(Obj->*(&Method)) != (void*)(&Method))



namespace nlpp
{

namespace poly
{

template <class V = Vec>
struct Function
{
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

namespace impl
{

template <class>
struct Function;


template <class V>
struct Function<poly::Function<V>>
{
    using F = poly::Function<V>;
    using Float = typename F::Float;


    Function(F* f) : f(f)
    {
        init();
    }

    Function(const Function& function) : f(function.f ? function.f->clone() : nullptr)
    {
        init();
    }

    Function& operator = (const Function& function)
    {
        if(function.f)
            f = std::unique_ptr<F>(function.f->clone());

        init();

        return *this;
    }


    void init ()
    {
        if(!f)
            return;

        if(NLPP_CHECK_IF_OVERRIDEN(F::functionGradient<, f))
    }


    using FunctionType = Float (F::*)(const Eigen::Ref<const V>&);
    using GradientType_1 = V (F::*)(const Eigen::Ref<const V>&);
    
    
    
    std::function<V                   (const Eigen::Ref<const V>&)               > gradient_1;
    std::function<void                (const Eigen::Ref<const V>&, Eigen::Ref<V>)> gradient_2;
    std::function<std::pair<Float, V> (const Eigen::Ref<const V>&)               > funcGrad_1;
    std::function<Float               (const Eigen::Ref<const V>&, Eigen::Ref<V>)> funcGrad_2; 



    std::unique_ptr<F> f;

    std::function<Float               (const Eigen::Ref<const V>&)               > function;
    std::function<V                   (const Eigen::Ref<const V>&)               > gradient_1;
    std::function<void                (const Eigen::Ref<const V>&, Eigen::Ref<V>)> gradient_2;
    std::function<std::pair<Float, V> (const Eigen::Ref<const V>&)               > funcGrad_1;
    std::function<Float               (const Eigen::Ref<const V>&, Eigen::Ref<V>)> funcGrad_2;

};



} // namespace impl


} // namespace nlpp