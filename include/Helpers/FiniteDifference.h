/** @file
    
    @brief Finite differencing calculation for scalar multivariate functions

    @details Simple and efficient automatic calculations of derivatives by using finite difference.
             Also, very easy to use:

    @snippet Helpers/FiniteDifference.cpp FiniteDifference class snippet
**/



#pragma once

#include "Helpers.h"


/** @defgroup FiniteDifferenceGroup Finite Difference
    @copydoc FiniteDifference.h
*/

//@{

/// Base definitions for CRTP
#define USING_FINITE_DIFFERENCE(...) using Base = __VA_ARGS__;  \
                                     using Base::Base;          \
                                     using Base::gradient;      \
                                     using Base::hessian;       \
                                     using Base::directional;   \
                                     using Base::f;             \
                                     using Base::step;

namespace cppnlp
{

/// Finite Difference namespace
namespace fd
{


template <typename>
struct SimpleStep;

template <typename>
struct NormalizedStep;


template <class, template <typename> class, typename>
struct Forward;

template <class, template <typename> class, typename>
struct Backward;

template <class, template <typename> class, typename>
struct Central;



namespace traits
{

template <class>
struct FiniteDifference;

template <class _Function, template <typename> class _Step, typename _Float>
struct FiniteDifference<Forward<_Function, _Step, _Float>>
{
    using Function = _Function;
    using Float = _Float;
    using Step = _Step<Float>;
};

template <class _Function, template <typename> class _Step, typename _Float>
struct FiniteDifference<Backward<_Function, _Step, _Float>>
{
    using Function = _Function;
    using Float = _Float;
    using Step = _Step<Float>;
};


template <class _Function, template <typename> class _Step, typename _Float>
struct FiniteDifference<Central<_Function, _Step, _Float>>
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

    template <typename X>
    auto gradient (const X& x) const
    {
        return static_cast<const Impl&>(*this).gradient(x, f(x));
    }

    template <class Derived>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).gradient(x, e, f(x));
    }


    template <typename X>
    auto hessian (const X& x) const
    {
        return static_cast<const Impl&>(*this).hessian(x, f(x));
    }

    template <class Derived>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).hessian(x, e, f(x));
    }


    template <typename X, typename E>
    auto directional (const X& x, const E& e) const
    {
        return directional(x, e, f(x));
    }

    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, const Fx& fx) const
    {
        return static_cast<const Impl&>(*this).directional(x, e, fx, step(f, x));
    }



    Function f;

    Step step;
};





template <class Function, template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Forward : public FiniteDifference<Forward<Function, Step, Float>>
{
    USING_FINITE_DIFFERENCE(FiniteDifference<Forward<Function, Step, Float>>);


    template <typename Fx>
    auto gradient (Float x, Fx fx) const
    {
        Float h = step(f, x);

        return directional(x, 1.0, fx, h);
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float h = step(f, x);
        Float temp;

        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> ret(x.rows(), x.cols());

        changeEval([&](const auto& x, int i){ ret(i) = (this->f(x) - fx) / h; }, x, h);

        return ret;
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, Fx fx) const
    {
        return directional(x, e, fx);
    }


    template <typename Fx>
    auto hessian (Float x, Fx fx) const
    {
        Float h = step(f, x);

        return (f(x - h) - 2 * fx + f(x + h)) / (h * h);
    }
    
    template <class Derived, typename Fx, std::enable_if_t<std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float temp1, temp2;
        Float h = step(f, x);
        Float h2 = h * h;

        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> fxi(x.rows(), x.cols());
        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        changeEval([&](const auto& x, int i){ fxi(i) = this->f(x); }, x, h);

        changeEval([&](const auto& x, int i)
        {
            changeEval([&](const auto& y, int j)
            {
                hess(i, j) = hess(j, i) = (this->f(y) - fxi(i) - fxi(j) + fx) / h2;
                
            }, x, h, handy::range(0, x.size()));
        }, x, h);

        return hess;
    }

    template <class Derived, class Fx, std::enable_if_t<!std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Fx& fx) const
    {
        Float h = step(f, x);

        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        changeEval([&](const auto& x, int i){ hess.col(i) = (this->f(x) - fx) / h; }, x, h);

        return hess;
    }

    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, const Fx& fx) const
    {
        return directional(x, e, fx);
    }



    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, const Fx& fx, Float h) const
    {
        return (f(x + h * e) - fx) / h;
    }


    template <class F, class Derived>
    static void changeEval (F f, const Eigen::MatrixBase<Derived>& x, typename Derived::Scalar inc)
    {
        changeEval(f, x, inc, handy::range(x.size()));
    }

    template <class F, class Derived, typename Int>
    static void changeEval (F f, const Eigen::MatrixBase<Derived>& x, typename Derived::Scalar inc, handy::Range<Int> range)
    {
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> y = x;

        for(auto i : range)
        {
            y(i) = x(i) + inc;

            f(y, i);

            y(i) = x(i);
        }
    }
};





template <class Function, template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Backward : public FiniteDifference<Backward<Function, Step, Float>>
{
    USING_FINITE_DIFFERENCE(FiniteDifference<Backward<Function, Step, Float>>);


    template <typename Fx>
    auto gradient (Float x, Fx fx) const
    {
        Float h = step(f, x);

        return directional(x, 1.0, fx, h);
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float h = step(f, x);
        Float temp;

        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> ret(x.rows(), x.cols());

        changeEval([&](const auto& x, int i){ ret(i) = (fx - this->f(x)) / h; }, x, h);

        return ret;
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, Fx fx) const
    {
        return directional(x, e, fx);
    }


    template <typename Fx>
    auto hessian (Float x, Fx fx) const
    {
        Float h = step(f, x);

        return (f(x - h) - 2 * fx + f(x + h)) / (h * h);
    }
    
    template <class Derived, typename Fx, std::enable_if_t<std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float temp1, temp2;
        Float h = step(f, x);
        Float h2 = h * h;

        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> fxi(x.rows(), x.cols());
        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        changeEval([&](const auto& x, int i){ fxi(i) = this->f(x); }, x, h);

        changeEval([&](const auto& x, int i)
        {
            changeEval([&](const auto& y, int j)
            {
                hess(i, j) = hess(j, i) = (fx - fxi(i) - fxi(j) + this->f(y)) / h2;

            }, x, h, handy::range(i, x.size()));
        }, x, h);

        return hess;
    }

    template <class Derived, class Fx, std::enable_if_t<!std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Fx& fx) const
    {
        Float h = step(f, x);

        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        changeEval([&](const auto& x, int i){ hess.col(i) = (this->f(x) - fx) / h; }, x, h);

        return hess;
    }

    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, const Fx& fx) const
    {
        return directional(x, e, fx);
    }



    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, const Fx& fx, Float h) const
    {
        return (fx - f(x - h * e)) / h;
    }



    template <class F, class Derived>
    static void changeEval (F f, const Eigen::MatrixBase<Derived>& x, typename Derived::Scalar inc)
    {
        changeEval(f, x, inc, handy::range(x.size()));
    }

    template <class F, class Derived, typename Int>
    static void changeEval (F f, const Eigen::MatrixBase<Derived>& x, typename Derived::Scalar inc, handy::Range<Int> range)
    {
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> y = x;

        for(auto i : range)
        {
            y(i) = x(i) - inc;

            f(y, i);

            y(i) = x(i);
        }
    }
};








template <class Function, template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Central : public FiniteDifference<Central<Function, Step, Float>>
{
    USING_FINITE_DIFFERENCE(FiniteDifference<Central<Function, Step, Float>>);

    
    auto gradient (Float x) const
    {
        return directional(x, 1.0);
    }

    template <typename Fx>
    auto gradient (Float x, Fx) const
    {
        return gradient(x);
    }

    

    template <class Derived>
    auto gradient (const Eigen::MatrixBase<Derived>& x) const
    {
        return gradient(x, step(f, x));
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        return gradient(x, fx, step(f, x));
    }



    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx, Float h) const
    {
        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> y = x;
        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> g(x.rows(), x.cols());

        for(int i = 0; i < x.size(); ++i)
        {
            y(i) = x(i) - h;

            auto fl = f(y);

            y(i) = x(i) + h;

            g(i) = (f(y) - fl) / (2 * h);

            y(i) = x(i);
        }

        return g;
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, Fx) const
    {
        return directional(x, e);
    }



    template <typename Fx>
    auto hessian (Float x, Fx fx) const
    {
        Float h = step(f, x);

        return (f(x + 2 * h) - 2 * fx + f(x - 2 * h)) / (4 * h * h);
    }

    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> y = x;
        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        Float h = step(f, x);
        Float h2 = 4 * h * h;
        
        for(int i = 0; i < x.size(); ++i)
        {
            y(i) = x(i) + 2 * h;

            auto fl = f(y);

            y(i) = x(i) - 2 * h;

            hess(i, i) = (f(y) - 2 * fx + fl) / h2;

            for(int j = i+1; j < x.size(); ++j)
            {
                y(i) = x(i) + h;
                y(j) = x(j) - h;

                auto frl = f(y);
                
                y(i) = x(i) - h;

                auto fll = f(y);

                y(j) = x(j) + h;

                auto flr = f(y);

                y(i) = x(i) + h;

                //handy::print(f(y), frl, flr, fll);

                hess(i, j) = hess(j, i) = (f(y) - frl - flr + fll) / h2;

                y(i) = x(i);
                y(j) = x(j);
            }
        }
        
        return hess;
    }


    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, Fx fx) const
    {
        return directional(x, e, fx);
    }


    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, Fx fx) const
    {
        return directional(x, e, fx, step(f, x));
    }

    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, Fx, Float h) const
    {
        return (f(x + h * e) - f(x - h * e)) / (2 * h);
    }
};






template <class Function, template <class, template <typename> class, typename> class Difference = Central,
          template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Gradient : public Difference<Function, Step, Float>
{
    USING_FINITE_DIFFERENCE(Difference<Function, Step, Float>);


    template <typename... Args>
    auto operator () (Args&&... args) const
    {
        return gradient(std::forward<Args>(args)...);
    }
};


template <class Function, template <class, template <typename> class, typename> class Difference = Central,
          template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Hessian : public Difference<Function, Step, Float>
{
    USING_FINITE_DIFFERENCE(Difference<Function, Step, Float>);

    Hessian (const Function& f) : Hessian(f, std::pow(constants::eps_<Float>, 3.0 / 4)) {}

    Hessian (const Function& f, const Step<Float>& step) : Base(f, step) {}


    template <typename... Args>
    auto operator () (Args&&... args) const
    {
        return hessian(std::forward<Args>(args)...);
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

    template <class Function, typename X>
    Float operator () (Function, const X& x) const
    {
        Float step = h * x.norm();

        if(step <= minValue)
            step = constants::eps_<Float>;

        return step;
    }

    Float operator () () const
    {
        return h;
    }


    static constexpr Float minValue = constants::eps_<Float> * constants::eps_<Float>;


    Float h;

    Float step;
};



template <template <class, template <typename> class, typename> class Difference = Central,
          class Function = void, template <typename> class Step = SimpleStep, typename Float = types::Float>
auto gradient (const Function& f)
{
    return Gradient<Function, Difference, Step, Float>(f);
}

template <template <class, template <typename> class, typename> class Difference = Central,
          class Function = void, template <typename> class Step = SimpleStep, typename Float = types::Float>
auto gradient (const Function& f, const Step<Float>& step)
{
    return Gradient<Function, Difference, Step, Float>(f, step);
}


template <template <class, template <typename> class, typename> class Difference = Central,
          class Function = void, template <typename> class Step = SimpleStep, typename Float = types::Float>
auto hessian (const Function& f)
{
    return Hessian<Function, Difference, Step, Float>(f);
}

template <template <class, template <typename> class, typename> class Difference = Central,
          class Function = void, template <typename> class Step = SimpleStep, typename Float = types::Float>
auto hessian (const Function& f, const Step<Float>& step)
{
    return Hessian<Function, Difference, Step, Float>(f, step);
}

//@}

} // namespace fd

} // namespace cppnlp