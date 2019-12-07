#pragma once

#include "helpers/helpers.hpp"
#include "utils/finiteDifference_dec.hpp"


namespace nlpp::wrap
{

namespace impl
{

template <class Impl, class V>
static constexpr bool isFunction = std::is_floating_point_v<OperatorType<Function<Impl>, V>>;

template <class Impl, class V>
static constexpr bool isGradient_1 = std::is_same_v<OperatorType<Impl, V, V&>, void>;

template <class Impl, class V>
static constexpr bool isGradient_2 = isMat<OperatorType<Impl, V>;

template <class Impl, class V>
static constexpr bool isFuncGrad_0 = std::is_floating_point_v<OperatorType<Impl, V, V&, bool>>;

template <class Impl, class V>
static constexpr bool isFuncGrad_1 = std::is_floating_point_v<OperatorType<Impl, V, V&>>;

template <class Impl, class V>
static constexpr bool isFuncGrad_2 = std::is_floating_point_v<std::tuple_element_t<0, OperatorType<Impl, V>>> &&
                                     isMat<std::tuple_element_t<1, OperatorType<Impl, V>>>;

template <class Impl, class V, class U>
static constexpr bool isHessian_1 = isMat<OperatorType<Impl, V, U>>;

template <class Impl, class V>
static constexpr bool isHessian_2 = isMat<OperatorType<Impl, V>> && OperatorType<Impl, V>::ColsAtCompileTime != 1;


template <class Impl>
Function<Impl>::Function (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::function (const Eigen::MatrixBase<V>& x)
{
    static_assert(isFunction<Impl, V>, "The functor has no interface for the given parameter")

    return Impl::operator()(x);
}

template <class Impl>
template <class V>
Scalar<V> Function<Impl>::operator () (const Eigen::MatrixBase<V>& x)
{
    return function(x);
}


template <class Impl>
Gradient<Impl>::Gradient (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
void Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    if constexpr(isGradient_1<Impl, V>)
        Impl::operator()(x, g);

    else if constexpr(isGradient_2<Impl, V>)
        g = Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter")
}
template <class Impl>
template <class V>
Plain<V> Gradient<Impl>::gradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isGradient_1<Impl, V>)
    {
        Plain<V> g(x.rows());
        Impl::operator(x, g);
        return g;
    }

    else if constexpr(isGradient_2<Impl, V>)
        return Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter")
}

template <class Impl>
template <class V>
void Gradient<V>::operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    gradient(x, g);
}

template <class Impl>
template <class V>
Plain<V> Gradient<V>::operator() (const Eigen::MatrixBase<V>& x)
{
    return gradient(x);
}


template <class Func, class Grad>
template <class F = Func, class G = Grad, std::enable_if_t<handy::HasConstructor<G>::value, int> = 0>
FunctionGradient<Func, Grad>::FunctionGradient (const F& f, const G& g) : Function<Func>(f), Gradient<Grad>(g) {}

template <class Func, class Grad>
template <class F = Func, class G = Grad, std::enable_if_t<!handy::HasConstructor<G>::value, int> = 0>
FunctionGradient<Func, Grad>::FunctionGradient (const F& f, const G& g) : Function<Func>(f), Gradient<Grad>(g) {}

template <class Func, class Grad>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, Grad>::functionGradient (const Eigen::MatrixBase<V>& x)
{
    return {function(x), gradient(x)};
}

template <class Func, class Grad>
template <class V>
Scalar<V> FunctionGradient<Func, Grad>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    if(calcGrad)
        gradient(x, g);

    return function(x);
}

template <class Func, class Grad>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, Grad>::operator() (const Eigen::MatrixBase<V>& x)
{
    return functionGradient(x);
}

template <class Func, class Grad>
template <class V>
Scalar<V> FunctionGradient<Func, Grad>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    return functionGradient(x, g, calcGrad);
}


template <class Func, template <class, class> class Difference, class Step>
FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::FunctionGradient (const Function& f) : Function(f), Gradient(f) {} 

template <class Func, template <class, class> class Difference, class Step>
FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::FunctionGradient (const Function& f, const Gradient& g) : Function(f), Gradient(Function(f)) {}

template <class Func, template <class, class> class Difference, class Step>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::functionGradient (const Eigen::MatrixBase<V>& x)
{
    auto f = function(x);
    return {f, gradient(x, f)};
}

template <class Func, template <class, class> class Difference, class Step>
template <class V>
Scalar<V> FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    auto f = function(x);

    if(calcGrad)
        gradient(x, g, f);

    return f;
}

template <class Func, template <class, class> class Difference, class Step>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::operator() (const Eigen::MatrixBase<V>& x)
{
    return functionGradient(x);
}

template <class Func, template <class, class> class Difference, class Step>
template <class V>
Scalar<V> FunctionGradient<Func, fd::Gradient<Func, Difference, Step>>::operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    return functionGradient(x, g, calcGrad);
}


template <class Impl>
FunctionGradient<Impl>::FunctionGradient (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, bool calcGrad)
{
    if constexpr(isFunctionGradient_1<Impl, V>)
        return Impl::operator()(x, g, calcGrad);

    else if constexpr(isFunctionGradient_2<Impl, V>)
        return Impl::operator()(x, g);

    else if constexpr(isFunctionGradient_3<Impl, V>)
    {
        Scalar<V> f;
        std::tie(f, g) = Impl::operator(x);
        return f;
    }

    else if constexpr(isFunction<Impl, V> && (isGradient_1<Impl, V> || isGradient_2<Impl, V>))
    {
        if(calcGrad)
            Gradient<Impl>(*this)(x, g);

        return Function<Impl>(*this)(x);
    }

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
std::pair<Scalar<V>, Plain<V>> FunctionGradient<Impl>::functionGradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isFunctionGradient_1<Impl, V> || isFunctionGradient_2<Impl, V>)
    {
        Plain<V> g(x.rows());
        Scalar<V> f = Impl::operator()(x, g);

        return {f, g};
    }

    else if constexpr(isFunctionGradient_3<Impl, V>)
        return Impl::operator()(x);

    else if constexpr(isFunction<Impl, V> && (isGradient_1<Impl, V> || isGradient_2<Impl, V>))
        return {Function<Impl>(*this)(x), Gradient<Impl>(*this)(x, g)};

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Scalar<V> FunctionGradient<Impl>::function (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isFunction<Impl, V>)
        return Function<Impl>(*this)(x);

    else
        return functionGrad(x);
}

template <class Impl>
template <class V>
void FunctionGradient<Impl>::gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g)
{
    if constexpr(isGradient_1<Impl, V> || isGradient_2<Impl, V>)
        Gradient<Impl>(*this)(x, g);

    else
        functionGrad(x);
}

template <class Impl>
template <class V>
Plain<V> FunctionGradient<Impl>::gradient (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isGradient_1<Impl, V> || isGradient_2<Impl, V>)
        return Gradient<Impl>(*this)(x);

    else
        return functionGrad(x);
}


template <class Impl>
Hessian::Hessian (const Impl& impl) : Impl(impl) {}

template <class Impl>
template <class V, class U>
Plain<V> Hessian::hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
{
    if constexpr(isHessian_1<Impl, V, U>)
        return Impl::operator()(x, e);

    else if constexpr(isHessian_2<Impl, V>)
        return Impl::operator()(x) * e;

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V>
Plain2D<V> Hessian::hessian (const Eigen::MatrixBase<V>& x)
{
    if constexpr(isHessian_1<Impl, V, U>)
    {
        Plain<V> e = Plain<V>::Constant(x.rows(), Scalar<V>{0.0});
        Plain2D<V> m(x.rows(), x.rows());

        for(int i = 0; i < e.rows(); ++i)
        {
            e(i) = Scalar<V>{1.0};
            m.row(i) = Impl::operator()(x, e);
            e(i) = Scalar<V>{0.0};
        }

        return m;
    }

    else if constexpr(isHessian_2<Impl, V>)
        return Impl::operator()(x);

    else
        static_assert(always_false<V>, "The functor has no interface for the given parameter");
}

template <class Impl>
template <class V, class U>
Plain<V> Hessian::operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e)
{
    return hessian(x, e);
}

template <class Impl>
template <class V>
Plain2D<V> Hessian::operator() (const Eigen::MatrixBase<V>& x)
{
    return hessian(x);
}

} // namespace impl




namespace poly
{

template <class V_ = Vec>
struct Function
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

    using FuncType = Float (const V&);


    Function () {}

    Function(const std::function<FuncType>& func) : func(func)
    {
    }

    Float function (const V& x)
    {
        return func(x);
    }

    Float operator () (const V& x)
    {
        return function(x);
    }

    std::function<FuncType> func;
};


template <class V_ = Vec>
struct Gradient
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

    using GradType_1 = V (const V&);
    using GradType_2 = void (const V&, ::nlpp::impl::Plain<V>&);


    Gradient () {}

    Gradient (const std::function<GradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : grad_1(grad_1)
    {
        init();
    }

    Gradient (const std::function<GradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : grad_2(grad_2)
    {
        init();
    }


    template <class G>
    Gradient (const G& grad) : Gradient(grad, ::nlpp::impl::Precedence<0>{}) {}



    V gradient (const V& x)
    {
        return grad_1(x);
    }

    void gradient (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        grad_2(x, g);
    }


    V operator () (const V& x)
    {
        return gradient(x);
    }

    void operator () (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        gradient(x, g);
    }


    Float directional (const V& x, const V& e, Float fx)
    {
        return direc(x, e, fx);
    }


    void init ()
    {
        if(!grad_1)
        {
            grad_1 = [gradImpl = nlpp::wrap::gradient(grad_2)](const V& x) mutable -> V
            {
                return gradImpl(x);
            };
        }

        if(!grad_2)
        {
            grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, ::nlpp::impl::Plain<V>& g) mutable
            {
                gradImpl(x, g);
            };
        }
    }


    std::function<GradType_1> grad_1;
    std::function<GradType_2> grad_2;
};


template <class V_ = Vec>
struct FunctionGradient : public Function<V_>, public Gradient<V_>
{
    using V = V_;
    using Float = ::nlpp::impl::Scalar<V>;

    using Func = Function<V>;
    using Grad = Gradient<V>;

    using Func::function;
    using Func::func;
    using Grad::gradient;
    using Grad::grad_1;
    using Grad::grad_2;

    using FuncType = typename Func::FuncType;
    using GradType_1 = typename Grad::GradType_1;
    using GradType_2 = typename Grad::GradType_2;

    using FuncGradType_1 = std::pair<Float, V> (const V&);
    using FuncGradType_2 = Float (const V&, ::nlpp::impl::Plain<V>&);
    
    using DirectionalType = Float (const V&, const V&, Float);


    FunctionGradient (const std::function<FuncGradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : funcGrad_1(funcGrad_1)
    {
        init();
    }

    FunctionGradient (const std::function<FuncGradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : funcGrad_2(funcGrad_2)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, const std::function<GradType_1>& grad_1, ::nlpp::impl::Precedence<0>) : Func(func), Grad(grad_1)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, const std::function<GradType_2>& grad_2, ::nlpp::impl::Precedence<1>) : Func(func), Grad(grad_2)
    {
        init();
    }

    FunctionGradient (const std::function<FuncType>& func, ::nlpp::impl::Precedence<0>) : Func(func)
    {
        init();
    }


    template <class F, std::enable_if_t<(::nlpp::wrap::FunctionType<F>::value >= 0) || (::nlpp::wrap::FunctionGradientType<F>::value >= 0), int> = 0>
    FunctionGradient(const F& func) : FunctionGradient(func, ::nlpp::impl::Precedence<0>{}) {}
    
    template <class F, class G, std::enable_if_t<(::nlpp::wrap::FunctionType<F>::value >= 0) && (::nlpp::wrap::GradientType<G>::value >= 0), int> = 0>
    FunctionGradient(const F& func, const G& grad) : FunctionGradient(func, grad, ::nlpp::impl::Precedence<0>{}) {}



    std::pair<Float, V> functionGradient (const V& x)
    {
        return funcGrad_1(x);
    }

    Float functionGradient (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        return funcGrad_2(x, g);
    }


    std::pair<Float, V> operator () (const V& x)
    {
        return functionGradient(x);
    }

    Float operator () (const V& x, ::nlpp::impl::Plain<V>& g)
    {
        return functionGradient(x.eval(), g);
    }


    Float directional (const V& x, const V& e)
    {
        return directional(x, e, func(x));
    }

    Float directional (const V& x, const V& e, Float fx)
    {
        return direc(x, e, fx);
    }


    template <class FuncGradImpl>
    void setFuncGrad(FuncGradImpl funcGradImpl)
    {
        setFuncGrad_1(funcGradImpl);
        setFuncGrad_2(funcGradImpl);
    }

    template <class FuncGradImpl>
    void setFuncGrad_1(FuncGradImpl funcGradImpl)
    {
        funcGrad_1 = [funcGradImpl](const V& x) mutable -> std::pair<Float, V>
            {
                return funcGradImpl(x);
            };
    }
    
    template <class FuncGradImpl>
    void setFuncGrad_2(FuncGradImpl funcGradImpl)
    {
        funcGrad_2 = [funcGradImpl](const V& x, ::nlpp::impl::Plain<V>& g) mutable -> Float
        {
            return funcGradImpl(x, g);
        }; 
    }


    void init ()
    {
        if(!funcGrad_1 && !funcGrad_2)
        {
            if(grad_1)
                setFuncGrad(::nlpp::wrap::functionGradient(func, grad_1));

            else if(grad_2)
                setFuncGrad(::nlpp::wrap::functionGradient(func, grad_2));

            else
                setFuncGrad(::nlpp::wrap::functionGradient(func));
        }

        else if(funcGrad_1 && !funcGrad_2)
            setFuncGrad_1(::nlpp::wrap::functionGradient(funcGrad_1));

        else if(!funcGrad_1 && funcGrad_2)
            setFuncGrad_2(::nlpp::wrap::functionGradient(funcGrad_2));

        if(!func)
        {
            if(funcGrad_1)
                func = [funcGrad = funcGrad_1](const V& x) -> Float
                {
                    return std::get<0>(funcGrad(x));
                };

            else if(funcGrad_2)
                func = [funcGrad = funcGrad_2](const V& x) -> Float
                {
                    V g(x.rows(), x.cols());
                    return funcGrad(x, g);
                };
        }

        if(!grad_1)
        {
            if(grad_2)
                grad_1 = [gradImpl = nlpp::wrap::gradient(grad_2)](const V& x) mutable -> V
                {
                    return gradImpl(x);
                };
            
            else
                grad_1 = [gradImpl = funcGrad_1](const V& x) mutable -> V
                {
                    return std::get<1>(gradImpl(x));
                };
        }
        
        if(!grad_2)
        {
            if(grad_1)
                grad_2 = [gradImpl = nlpp::wrap::gradient(grad_1)](const V& x, ::nlpp::impl::Plain<V>& g) mutable
                {
                    gradImpl(x, g);
                };
            
            else
                grad_2 = [gradImpl = funcGrad_2](const V& x, ::nlpp::impl::Plain<V>& g) mutable
                {
                    gradImpl(x, g);
                };
        }

        direc = [funcGrad = ::nlpp::wrap::functionGradient(func)](const V& x, const V& e, Float fx) mutable -> Float
        {
            return funcGrad.directional(x, e, fx);
        };
    }



    std::function<FuncGradType_1> funcGrad_1;
    std::function<FuncGradType_2> funcGrad_2;
    std::function<DirectionalType> direc;
};


template <class V_ = ::nlpp::Vec, class M_ = ::nlpp::Mat>
struct Hessian
{
    using V = V_;
    using M = M_;
    using Float = ::nlpp::impl::Scalar<V>;

    using HessType = M (const Eigen::MatrixBase<V>&);

    Hessian (const std::function<HessType>& hessian) : hessian(hessian) {}


    M operator () (const V& x)
    {
        return hessian(x);
    }


    std::function<HessType> hessian;
};



} // namespace poly

} // namespace nlpp::wrap