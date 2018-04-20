#pragma once

#include "Helpers.h"


#define USING_FINITE_DIFFERENCE(...) using Base = __VA_ARGS__;  \
                                     using Base::Base;          \
                                     using Base::gradient;      \
                                     using Base::hessian;       \
                                     using Base::f;             \
                                     using Base::step;

namespace cppnlp
{

template <typename>
struct SimpleStep;

template <typename>
struct NormalizedStep;


template <class, template <typename> class, typename>
struct ForwardDifference;




namespace impl
{

template <class Function, class Scalar, std::enable_if_t<std::is_fundamental<std::decay_t<Scalar>>::value, int> = 0>
void changeEval (Function f, const Scalar& x, Scalar val)
{
    auto temp = x;

    const_cast<Scalar&>(x) = val;

    f();

    const_cast<Scalar&>(x) = temp;
}

} // namespace impl



namespace traits
{

template <class>
struct FiniteDifference;

template <class _Function, template <typename> class _Step, typename _Float>
struct FiniteDifference<ForwardDifference<_Function, _Step, _Float>>
{
    using Function = _Function;
    using Float = _Float;
    using Step = _Step<Float>;
};

} // namespace traits



template <class Impl>
struct FiniteDifference
{
    using Function = typename traits::FiniteDifference<Impl>::Function;
    using Step = typename traits::FiniteDifference<Impl>::Step;
    using Float = typename traits::FiniteDifference<Impl>::Float;


    FiniteDifference (const Function& f, const Step& step = Step{}) : f(f), step(step)
    {
    }

    template <class Derived>
    auto gradient (const Eigen::MatrixBase<Derived>& x) const
    {
        return static_cast<const Impl&>(*this).gradient(x, f(x));
    }

    template <class Derived>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).gradient(x, e, f(x));
    }


    template <class Derived>
    auto hessian (const Eigen::MatrixBase<Derived>& x) const
    {
        return static_cast<const Impl&>(*this).hessian(x, f(x));
    }

    template <class Derived>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).hessian(x, e, f(x));
    }


    Function f;

    Step step;
};





template <class Function, template <typename> class Step = SimpleStep, typename Float = types::Float>
struct ForwardDifference : public FiniteDifference<ForwardDifference<Function, Step, Float>>
{
    USING_FINITE_DIFFERENCE(FiniteDifference<ForwardDifference<Function, Step, Float>>);


    template <class Derived>
    Derived gradient (const Eigen::MatrixBase<Derived>& x, std::result_of_t<Function(Derived)> fx) const
    {
        Float h = step(f, x);
        Float temp;

        return Derived::NullaryExpr(x.rows(), x.cols(), [&](int i)
        {
            impl::changeEval([&]{ temp = (f(x) - fx) / h; }, x(i), x(i) + h);

            return temp;
        });
    }

    template <class Derived>
    std::result_of_t<Function(Derived)> gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e,
                                                  std::result_of_t<Function(Derived)> fx) const
    {
        return directional(x, e, fx);
    }
    
    
    template <class Derived, std::enable_if_t<std::is_fundamental<std::decay_t<std::result_of_t<Function(Derived)>>>::value, int> = 0>
    MatX<Float> hessian (const Eigen::MatrixBase<Derived>& x, std::result_of_t<Function(Derived)> fx) const
    {
        MatX<Float> hess(x.size(), x.size());

        Float temp1, temp2;
        Float h = step(f, x);
        Float h2 = h * h;

        Derived fxi = Derived::NullaryExpr(x.rows(), x.cols(), [&](int i)
        {
            impl::changeEval([&]{ temp1 = f(x); }, x(i), x(i) + h);

            return temp1;
        });
        

        for(int i = 0; i < x.size(); ++i)
        {
            impl::changeEval([&]
            {
                for(int j = i; j < x.size(); ++j)
                    impl::changeEval([&]{ hess(i, j) = (f(x) - fxi(i) - fxi(j) + fx) / h2; }, x(j), x(j) + h);

            }, x(i), x(i) + h);
        }


        return hess;
    }

    template <class Derived, std::enable_if_t<!std::is_fundamental<std::decay_t<std::result_of_t<Function(Derived)>>>::value, int> = 0>
    MatX<Float> hessian (const Eigen::MatrixBase<Derived>& x, const std::result_of_t<Function(Derived)>& fx) const
    {
        MatX<Float> hess(fx.rows(), x.rows());

        Float temp;
        Float h = step(f, x);

        for(int i = 0; i < hess.cols(); ++i)
            impl::changeEval([&]{ hess.col(i) = (f(x) - fx) / h; }, x(i), x(i) + h);

        return hess;
    }

    template <class Derived>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e,
                  const std::result_of_t<Function(Derived)>& fx) const
    {
        return directional(x, e, fx);
    }




    template <class Derived>
    auto directional (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return directional(x, e, f(x));
    }

    template <class Derived>
    auto directional (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, 
                             const std::result_of_t<Function(Derived)>& fx) const
    {
        return directional(x, e, fx, step(f, x));
    }

    template <class Derived>
    auto directional (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e,
                      const std::result_of_t<Function(Derived)>& fx, Float h) const
    {
        return (f(x + h * e) - fx) / h;
    }
};







template <class Function, template <class, template <typename> class, typename> class Difference = ForwardDifference,
          template <typename> class Step = SimpleStep, typename Float = types::Float>
struct GradientFD : public Difference<Function, Step, Float>
{
    USING_FINITE_DIFFERENCE(Difference<Function, Step, Float>);


    template <class Derived> 
    auto operator () (const Eigen::MatrixBase<Derived>& x) const
    {
        return gradient(x);
    }

    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const std::result_of_t<Function(Derived)>& fx) const
    {
        return gradient(x, fx);       
    }

    
    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return gradient(x, e);
    }

    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e,
                      const std::result_of_t<Function(Derived)>& fx) const
    {
        return gradient(x, e, fx);
    }
};


template <class Function, template <class, template <typename> class, typename> class Difference = ForwardDifference,
          template <typename> class Step = SimpleStep, typename Float = types::Float>
struct HessianFD : public Difference<Function, Step, Float>
{
    USING_FINITE_DIFFERENCE(Difference<Function, Step, Float>);

    HessianFD (const Function& f, const Step<Float>& step = { std::sqrt(constants::eps_<Float>) }) : Base(f, step) {}


    template <class Derived> 
    auto operator () (const Eigen::MatrixBase<Derived>& x) const
    {
        return hessian(x);
    }

    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const std::result_of_t<Function(Derived)>& fx) const
    {
        return hessian(x, fx);       
    }

    
    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return hessian(x, e);
    }

    template <class Derived>
    auto operator () (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e,
                      const std::result_of_t<Function(Derived)>& fx) const
    {
        return hessian(x, e, fx);
    }
};






template <typename Float = types::Float>
struct SimpleStep
{
    SimpleStep (Float h = constants::eps_<Float>) : h(h) {}

    Float operator () (...) const
    {
        return h;
    }
 
    Float h;
};


template <typename Float = types::Float>
struct NormalizedStep
{
    NormalizedStep (Float h = constants::eps_<Float>) : h(h) {}

    template <class Function, class Derived>
    Float operator () (Function, const Eigen::MatrixBase<Derived>& x) const
    {
        Float step = h * x.norm();

        if(step <= minValue)
            step = constants::eps_<Float>;

        return step;
    }

    template <class Function, class Derived>
    Float operator () () const
    {
        return h;
    }


    static constexpr Float minValue = constants::eps_<Float> * constants::eps_<Float>;


    Float h;

    Float step;
};




template <class Function, template <typename> class Step = SimpleStep,
          template <class, template <typename> class, typename> class Difference = ForwardDifference,
          typename Float = types::Float>
auto gradientFD (const Function& f)
{
    return GradientFD<Function, Difference, Step, Float>(f);
}

template <class Function, template <typename> class Step = SimpleStep,
          template <class, template <typename> class, typename> class Difference = ForwardDifference,
          typename Float = types::Float>
auto gradientFD (const Function& f, const Step<Float>& step)
{
    return GradientFD<Function, Difference, Step, Float>(f, step);
}


template <class Function, template <typename> class Step = SimpleStep,
          template <class, template <typename> class, typename> class Difference = ForwardDifference,
          typename Float = types::Float>
auto hessianFD (const Function& f)
{
    return HessianFD<Function, Difference, Step, Float>(f);
}

template <class Function, template <typename> class Step = SimpleStep,
          template <class, template <typename> class, typename> class Difference = ForwardDifference,
          typename Float = types::Float>
auto hessianFD (const Function& f, const Step<Float>& step)
{
    return HessianFD<Function, Difference, Step, Float>(f, step);
}


} // namespace cppnlp