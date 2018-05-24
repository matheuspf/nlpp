/** @file
    
    @brief Finite differencing calculation for scalar multivariable functions

    @details Simple and efficient automatic calculations of derivatives by using finite difference. It contains
             methods for forward, backward and central finite derivatives calculation. Take a look at this 
             gorgeous interface:

    @snippet FiniteDifference.cpp FiniteDifference snippet

    @details You can also determine the specific (infinitesimal) step size by using the @c Step template
             class. The default is simply a constant to every dimension: @f$\sqrt(u)@f$ for forward and
             central differences, and @f$u^\frac{2}{3}@f$ for central differences, where @f$u@f$ is the 
             machine precision constant (usually @f$1.1^{-16}@f$).

             The class must be a functor (has the @c operator()) receiving an Eigen::DenseBase matrix that
             returns a scalar value, the step size.

    @snippet Helpers/FiniteDifference.cpp FiniteDifference step snippet

    @note Notice that the functor is copied into the finite difference classes. So be careful about
          lifetime issues for your objects.
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


/** @name
    @brief Forward Declarations of the @c Step classes
*/
//@{
template <typename>
struct SimpleStep;

template <typename>
struct NormalizedStep;
//@}


/** @name
    @brief Forward declaration of the main finite difference classes
*/
//@{
template <class, template <typename> class, typename>
struct Forward;

template <class, template <typename> class, typename>
struct Backward;

template <class, template <typename> class, typename>
struct Central;
//@}



namespace traits
{

/** @name
    @brief These guys are needed because we need to access some derived types in the base CRTP class.
           Really boring and uggly stuff, but its essential.
*/
//@{
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
//@}

} // namespace traits



/** @brief The base finite difference class, used with CRTP
  
    @tparam Impl The base class that extends from FiniteDifference<Impl>. It must expose:
                 - @c Function: Functor type of which we want to take the derivatives of
                 - @c Step: Type of the infinitesinal step calculator functor
                 - @c Float: Floating type

*/
template <class Impl>
struct FiniteDifference
{
    /** @name
        @brief Base type definitions from @c Impl
    */
    //@{
    using Function = typename traits::FiniteDifference<Impl>::Function;
    using Step = typename traits::FiniteDifference<Impl>::Step;
    using Float = typename traits::FiniteDifference<Impl>::Float;
    //@}

    /** @brief Single base constructor
        @param f The functor type
        @param step Step size type
    */
    FiniteDifference (const Function& f, const Step& step = Step{}) : f(f), step(step)
    {
    }

    /** @brief Gradient evalutation. Uses CRTP, delegating the call to @c Impl with @c f(x) calculated.
        
        @param x The variable which we want to calculate the gradient of @c f. It can be both an
                     Eigen::MatrixBase or a scalar
        @returns If @c x is an Eigen::MatrixBase, returns an Eigen vector of same dimension as @c x. If @c x
                 has constant rows/cols, the return will have as well. If @c x is a scalar, returns a scalar
                 representing the derivative of @c f at @c x.
    */
    template <typename X>
    auto gradient (const X& x) const
    {
        return static_cast<const Impl&>(*this).gradient(x, f(x));
    }


    /** @copybrief gradient()
     * 
     *  @param x An Eigen::MatrixBase vector/matrix
     *  @param e The direction which we want to calculate a single step difference.
     * 
     *  @returns A scalar having the value of @f$\lim_{h\to0} \frac{f(x+e) - f(x)}{h}@f$, where
     *           @a h is given by @a step(x).
     * 
     *  @note Requirements:
     *        - x.rows() == e.rows()
     *        - x.cols() == e.cols()
    */
    template <class Derived>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).gradient(x, e, f(x));
    }


    /** @brief Hessian evalutation. Uses CRTP, delegating the call to @c Impl with @c f(x) calculated.
    
        @param x The variable which we want to calculate the hessian of @c f. It can be both an
                        Eigen::MatrixBase or a scalar
        @returns If @c x is an Eigen::MatrixBase, returns an Eigen matrix of dimensions @f$N \times N@f$, where
                @c N = @c x.size(). If @c x is a scalar, returns the second derivative of @c f at @c x.
    */
    template <typename X>
    auto hessian (const X& x) const
    {
        return static_cast<const Impl&>(*this).hessian(x, f(x));
    }

    /** @brief Hessian vector product calculation: 
    
        @param x The variable which we want to calculate the hessian vector product at.
        @param e The direction to be projected over the hessian matrix.
    
        @returns A vector of size @c e.size()

        @note Requirements:
              - x.rows() == x.cols()
              - x.rows() == e.rows()
              - e.cols() == 1
    */
    template <class Derived>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e) const
    {
        return static_cast<const Impl&>(*this).hessian(x, e, f(x));
    }

    /** @brief General directional derivative calculation @f$\nabla^2 f(x)^\intercal e$@f.
     *  
     *  @details Calculate the directional derivative of @c f at @c x in the direction of @c e. That is:
     *           # Forward: @f$\lim_{h\to0} \frac{f(x+e) - f(x)}{h}@f$
     *           # Backward: @f$\lim_{h\to0} \frac{f(x) - f(x-e)}{h}@f$
     *           # Central: @f$\lim_{h\to0} \frac{f(x+e) - f(x-e)}{2*h}@f$
     *  
     *  @param x Either an Eigen::MatrixBase or a scalar
     *  @param e Either an Eigen::MatrixBase or a scalar 
     *  
     *  @attention Requirements:
     *             # (@c x.size() == @c e.size()) OR (@c x and @c e are scalars )
     *          
    */
    template <typename X, typename E>
    auto directional (const X& x, const E& e) const
    {
        return directional(x, e, f(x));
    }


    ///@copydoc directional()
    template <typename X, typename E, typename Fx>
    auto directional (const X& x, const E& e, const Fx& fx) const
    {
        return static_cast<const Impl&>(*this).directional(x, e, fx, step(x));
    }


    Function f;     ///< Functor variable

    Step step;      ///< Step variable
};




/** @brief Forward difference class
 * 
 *  @details This is the actual implementation of the gradients and hessian finite differences for forward
 *           difference. The FiniteDifference class only delegates the call to this class via CRTP.
 * 
 *           Notice that for every function in this class @c f(x) is already calculated. It is because in
 *           forward difference its necessary to know the value of @c f at @c x. This also provides the user 
 *           a nice interface, where you can just reuse an already calculated function call.
 * 
 *  @tparam Function Functor type
 *  @tparam Step Step size template functor
 *  @tparam Float Base floating point type
*/
template <class Function, template <typename> class Step = SimpleStep, typename Float = types::Float>
struct Forward : public FiniteDifference<Forward<Function, Step, Float>>
{
    USING_FINITE_DIFFERENCE(FiniteDifference<Forward<Function, Step, Float>>);


    /** @name
     *  @brief Forward difference gradient routines
    */
    //@{
    /** @brief Derivative calculation for scalar types
     *  @param x Floating point value
     *  @param fx Scalar result of @c f(x)
     *  @returns @f$ \frac{\partial f}{\partial x} @f$
    */
    template <typename Fx>
    auto gradient (Float x, Fx fx) const
    {
        Float h = step(x);

        return directional(x, 1.0, fx, h);
    }

    /** @brief Gradient calculation for multivariable inputs
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$ \nabla f(x) @f$
    */
    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float h = step(x);
        Float temp;

        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> ret(x.rows(), x.cols());

        changeEval([&](const auto& x, int i){ ret(i) = (this->f(x) - fx) / h; }, x, h);

        return ret;
    }

    /** @brief Directional gradient calculation
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param e Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$\lim_{h\to0} \frac{f(x+e) - f(x)}{h}@f$
    */
    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, Fx fx) const
    {
        return directional(x, e, fx);
    }
    //@}


    /** @name
     *  @brief Forward difference hessian routines
    */
    //@{
    /** @brief Second derivative calculation for scalars
     *  @param x Floating point value
     *  @param fx Scalar result of @c f(x)
     *  @returns @f$ \frac{\partial f}{\partial x} @f$
    */
    template <typename Fx>
    auto hessian (Float x, Fx fx) const
    {
        Float h = step(x);

        return (f(x - h) - 2 * fx + f(x + h)) / (h * h);
    }

    /** @brief Hessian calculation for multivariable function at the point @c x
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$ \nabla^2 f(x) @f$
    */
    template <class Derived, typename Fx, std::enable_if_t<std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float temp1, temp2;
        Float h = step(x);
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

    /** @brief Hessian calculation for multivariable @a vector functions at point @c x
     * 
     *  @details This is useful when you have, for example, the exact gradient function of @c f. This way,
     *           we approximate the hessian of @c f by using N+1 gradient calls.
     * 
     *  @param x Eigen::MatrixBase vector/matrix
     *  @param fx Vector result of @c f(x)
     *  @returns @f$ \nabla f(x) @f$
     */
    template <class Derived, class Fx, std::enable_if_t<!std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Fx& fx) const
    {
        Float h = step(x);

        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        changeEval([&](const auto& x, int i){ hess.col(i) = (this->f(x) - fx) / h; }, x, h);

        return hess;
    }

    /** @brief Hessian vector product calculation at point @c x with direction @c e
     * 
     *  @details This is a very useful procedure, which costs a single function call (plus the call to @c f(x))
     *           to calculate a hessian vector product. Some algorithms only need this product instead of the
     *           full hessian.
     * 
     *  @param x Eigen::MatrixBase vector/matrix
     *  @param e Eigen::MatrixBase vector/matrix
     *  @param fx Scalar result of @c f(x)
     *  @returns @f$ \nabla^2 f(x) \intercal e @f$
     */
    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<Derived>& e, const Fx& fx) const
    {
        return directional(x, e, fx);
    }
    //@}


    /** @brief Forward finite approximation of the gradient of @c f
     * 
     *  @details This is a very general routine that can be used for both vector and scalar inputs.
     * 
     *  @param x A scalar or Eigen::MatrixBase matrix/vector
     *  @param e A scalar or Eigen::MatrixBase matrix/vector
     *  @param fx A scalar or Eigen::MatrixBase matrix/vector resulting from a call to @c f(x)
     *  @param h The infinitesimal step size
     *  @returns f(x)
    */
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
        Float h = step(x);

        return directional(x, 1.0, fx, h);
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float h = step(x);
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
        Float h = step(x);

        return (f(x - h) - 2 * fx + f(x + h)) / (h * h);
    }
    
    template <class Derived, typename Fx, std::enable_if_t<std::is_fundamental<std::decay_t<Fx>>::value, int> = 0>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Float temp1, temp2;
        Float h = step(x);
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
        Float h = step(x);

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
        return gradient(x, step(x));
    }

    template <class Derived, typename Fx>
    auto gradient (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        return gradient(x, fx, step(x));
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
        Float h = step(x);

        return (f(x + 2 * h) - 2 * fx + f(x - 2 * h)) / (4 * h * h);
    }

    template <class Derived, typename Fx>
    auto hessian (const Eigen::MatrixBase<Derived>& x, Fx fx) const
    {
        Eigen::Matrix<Float, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> y = x;
        Eigen::Matrix<Float, Derived::SizeAtCompileTime, Derived::SizeAtCompileTime> hess(x.size(), x.size());

        Float h = step(x);
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
        return directional(x, e, fx, step(x));
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

    template <typename X>
    Float operator () (const X& x, int i) const
    {
        Float step = h * std::abs(x[i]);

        if(step <= minValue)
            step = constants::eps_<Float>;

        return step;
    }

    template <typename X>
    Float operator () (const X&) const
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