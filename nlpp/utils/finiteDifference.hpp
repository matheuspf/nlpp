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

#include "helpers/helpers.hpp"
#include "utils/wrappers_dec.hpp"

/** @defgroup FiniteDifferenceGroup Finite Difference
    @copydoc FiniteDifference.h
*/
//@{

/// Base definitions for CRTP
#define CPPOPT_USING_FINITE_DIFFERENCE(NAME, ...) using NAME = __VA_ARGS__;  \
                                                  using NAME::NAME;          \
                                                  using NAME::gradient;      \
                                                  using NAME::hessian;       \
                                                  using NAME::gradientDir;   \
                                                  using NAME::hessianDir;   \
                                                  using NAME::f;             \
                                                  using NAME::step;

#define CPPOPT_USING_STEPSIZE(NAME, ...) using NAME = __VA_ARGS__;   \
                                         using NAME::NAME;           \
                                         using NAME::operator();     \
                                         using NAME::h;


namespace nlpp
{

/// Finite Difference namespace
namespace fd
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D;

/** @name
    @brief Forward Declarations of the @c Step classes
*/
//@{
struct AutoStep;

template <typename Float = types::Float>
struct SimpleStep;

template <typename = types::Float>
struct NormalizedStep;
//@}


/** @name
    @brief Forward declaration of the main finite difference classes
*/

template <class, class Step = AutoStep>
struct Forward;

template <class, class Step = AutoStep>
struct Backward;

template <class, class Step = AutoStep>
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

template <class _Function, class _Step>
struct FiniteDifference<Forward<_Function, _Step>>
{
    using Function = _Function;
    using Step = _Step;
};

template <class _Function, class _Step>
struct FiniteDifference<Backward<_Function, _Step>>
{
    using Function = _Function;
    using Step = _Step;
};

template <class _Function, class _Step>
struct FiniteDifference<Central<_Function, _Step>>
{
    using Function = _Function;
    using Step = _Step;
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
    //@{
    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x) const
    {
        return static_cast<const Impl&>(*this).gradient(x, f(x));
    }

    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        return static_cast<const Impl&>(*this).gradient(x, g, f(x));
    }

    template <class V, class U>
    auto gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return static_cast<const Impl&>(*this).gradient(x, e, f(x));
    }


    /** @brief Hessian evalutation. Uses CRTP, delegating the call to @c Impl with @c f(x) calculated.
    
        @param x The variable which we want to calculate the hessian of @c f. It can be both an
                        Eigen::MatrixBase or a scalar
        @returns If @c x is an Eigen::MatrixBase, returns an Eigen matrix of dimensions @f$N \times N@f$, where
                @c N = @c x.size(). If @c x is a scalar, returns the second derivative of @c f at @c x.
    */
    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        return static_cast<const Impl&>(*this).hessian(x, h, f(x));
    }

    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x) const
    {
        return static_cast<const Impl&>(*this).hessian(x, f(x));
    }

    template <class V, class U>
    auto hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return static_cast<const Impl&>(*this).hessian(x, e, gradient(x));
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

    template <class V, class U>
    auto gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return static_cast<const Impl&>(*this).gradientDir(x, e, f(x));
    }

    template <class V, class U>
    auto hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return static_cast<const Impl&>(*this).hessianDir(x, e, gradient(x));
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
template <class Function, class Step>
struct Forward : public FiniteDifference<Forward<Function, Step>>
{
    CPPOPT_USING_FINITE_DIFFERENCE(Base, FiniteDifference<Forward<Function, Step>>);


    /** @name
     *  @brief Forward difference gradient routines
    */
    //@{
    /** @brief Gradient calculation for multivariable inputs
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$ \nabla f(x) @f$
    */
    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x, Scalar<V> fx) const
    {
        Plain<V> g(x.rows(), x.cols());

        gradient(x, g, fx);

        return g;
    }

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V> fx) const
    {
        changeEval([&](const auto& x, int i, double h){ g(i) = (this->f(x) - fx) / h; }, x, step);
    }

    /** @brief Directional gradient calculation
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param e Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$\lim_{h\to0} \frac{f(x+e) - f(x)}{h}@f$
    */
    template <class V, class U>
    auto gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const
    {
        return gradientDir(x, e, fx);
    }
    //@}


    /** @name
     *  @brief Forward difference hessian routines
    */
    //@{
    /** @brief Hessian calculation for multivariable function at the point @c x
    *   @param x Eigen::MatrixBase vector/matrix
    *   @param fx Scalar result of @c f(x)
    *   @returns @f$ \nabla^2 f(x) @f$
    */
    template <class V>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& hess, Scalar<V> fx) const
    {
        using Float = Scalar<V>;

        Float temp1, temp2;
        Float h = step(x);
        Float h2 = h * h;

        Plain<V> fxi(x.rows(), x.cols());

        changeEval([&](const auto& x, int i, Float){ fxi(i) = this->f(x); }, x, h);

        changeEval([&](const auto& x, int i, Float)
        {
            changeEval([&](const auto& y, int j, Float)
            {
                hess(i, j) = hess(j, i) = (this->f(y) - fxi(i) - fxi(j) + fx) / h2;
                
            }, x, h, handy::range(i, x.size()));
        }, x, h);
    }

    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, Scalar<V> fx) const
    {
        Plain2D<V> hess(x.size(), x.size());
        hessian(x, hess, fx);

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
    template <class V, class U, class W>
    auto hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>& gx) const
    {
        return hessianDir(x, e, gx);
    }
    //@}


    /** @brief Forward finite approximation of the gradient of @c f
     * 
     *  @details This is a very general routine that can be used for both vector and scalar inputs.
     * 
     *  @param x A scalar or Eigen::MatrixBase matrix/vector
     *  @param e A scalar or Eigen::MatrixBase matrix/vector
     *  @param fx A scalar or Eigen::MatrixBase matrix/vector resulting from a call to @c f(x)
     *  @param h The infinitezimal step size
     *  @returns f(x)
    */
    template <class V, class U>
    auto gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const
    {
        auto h = step(x);
        return (f(x + h * e) - fx) / h;
    }

    template <class V, class U, class W>
    auto hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>& gx) const
    {
        auto h = step(x);
        return (gradient(x + h * e) - gx) / h;
    }


    /** @name
     *  @brief Calculate @f$f(x+inc*e_i)$@f for @f$i \in range$@f
     * 
     *  @details Useful for calculating the derivatives at @c x.
     * 
     *  @param F The functor we want to evaluate.
     *  @param x The vector to evaluate
     *  @param inc The total increment in each dimension
     *  @param range The range to be evaluated
     *  
     *  @note Default range is <tt>0, ..., x.size()</tt>
    */
    //@{
    template <class F, class V>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, Scalar<V> inc)
    {
        changeEval(f, x, [&](auto...){ return inc; });
    }

    template <class F, class V, class StepSize, std::enable_if_t<!std::is_fundamental<StepSize>::value, int> = 0>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, const StepSize& stepSize)
    {
        changeEval(f, x, stepSize, handy::range(x.size()));
    }

    template <class F, class V, typename Int>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, Scalar<V> inc, handy::Range<Int> range)
    {
        changeEval(f, x, [&](auto...){ return inc; }, range);
    }

    template <class F, class V, class StepSize, typename Int, std::enable_if_t<!std::is_fundamental<StepSize>::value, int> = 0>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, const StepSize& stepSize, handy::Range<Int> range)
    {
        Eigen::Matrix<Scalar<V>, V::RowsAtCompileTime, V::ColsAtCompileTime> y = x;

        for(auto i : range)
        {
            auto inc = stepSize(x, i);

            y(i) = x(i) + inc;

            f(y, i, inc);

            y(i) = x(i);
        }
    }
    //@}
};




/** @brief Backward difference class
 * 
 *  @details This is the actual implementation of the gradients and hessian finite differences for backward
 *           difference. Calls are delegated bie CRTP
 * 
 *  @tparam Function Functor type
 *  @tparam Step Step size template functor
 *  @tparam Float Base floating point type
*/
template <class Function, class Step>
struct Backward : public FiniteDifference<Backward<Function, Step>>
{
    CPPOPT_USING_FINITE_DIFFERENCE(Base, FiniteDifference<Backward<Function, Step>>);


    /** @name
     *  @brief Gradient calculation with backward difference
     * 
     *  @details Calculate the gradient of @c f at a given vector or scalar @c x, given a already calculated
     *           (via base CRTP class) function value @c fx = @c f(x). Also, given an direction @c e, calculate
     *           the directional derivative in the direction of @c e.
     * 
     *  @param x The scalar or Eigen::MatrixBase value where the gradient (or first derivative, if scalar) of @c f will be evaluated
     *  @param fx The result of @c f applied to @c x, that is, f(x)
     *  @param g A reference to where we will write the gradient of @c f calculated at @c x
     *  @param e The direction to calculate the directional derivative of @f c at @c x in the direction of @c e
    */
    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x, Scalar<V> fx) const
    {
        Plain<V> g(x.rows(), x.cols());

        gradient(x, g, fx);

        return g;
    }

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, impl::Plain<V>& g, Scalar<V> fx) const
    {
        changeEval([&](const auto& x, int i, double h){ static_cast<impl::Plain<V>>(g)(i) = (fx - this->f(x)) / h; }, x, step);
    }

    template <class V, class U>
    auto gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const
    {
        return gradientDir(x, e, fx);
    }
// 
    //@}



    /** @name
     *  @brief Hessian calculation for backward difference
     * 
     *  @details Calculate the hessian of @c f at a given vector or scalar @c x, given a already calculated
     *           (via base CRTP class) function value @c fx = @c f(x). Also, given an direction @c e, calculate
     *           the hessian product: @f$\nabla^2 f(x) e$@f.
     * 
     *  @param x The scalar or Eigen::MatrixBase value where the hessian (or second difference, if scalar) of @c f will be evaluated
     *  @param fx The result of @c f applied to @c x, that is, f(x). If it is a vector, we return the Jacobian of @c f.
     *  @param e The direction to calculate the hessian vector product @f$\nabla^2 f(x) e$@f.
    */
    //@{
    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& hess, Scalar<V> fx) const
    {
        using Float = Scalar<V>;

        Float temp1, temp2;
        Float h = step(x);
        Float h2 = h * h;

        Plain<V> fxi(x.rows(), x.cols());

        changeEval([&](const auto& x, int i, double){ fxi(i) = this->f(x); }, x, h);

        changeEval([&](const auto& x, int i, double)
        {
            changeEval([&](const auto& y, int j, double)
            {
                hess(i, j) = hess(j, i) = (fx - fxi(i) - fxi(j) + this->f(y)) / h2;

            }, x, h, handy::range(i, x.size()));
        }, x, h);
    }

    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, Scalar<V> fx) const
    {
        impl::Plain2D<V> hess(x.size(), x.size());
        hessian(x, hess, fx);

        return hess;
    }

    template <class V, class U, class W>
    auto hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>& gx) const
    {
        return hessianDir(x, e, gx);
    }
    //@}


    
    template <class V, class U>
    auto gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V> fx) const
    {
        auto h = step(x);
        return (fx - f(x - h * e)) / h;
    }

    template <class V, class U, class W>
    auto hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>& gx) const
    {
        auto h = step(x);
        return (gx - gradient(x - h * e)) / h;
    }



    /** @name
     *  @brief Calculate @f$f(x-inc*e_i)$@f for @f$i \in range$@f
     * 
     *  @details Useful for calculating the derivatives at @c x.
     * 
     *  @param F The functor we want to evaluate.
     *  @param x The vector to evaluate
     *  @param inc The total increment in each dimension
     *  @param range The range to be evaluated
     *  
     *  @note Default range is <tt>0, ..., x.size()</tt>
    */
    //@{
    template <class F, class V>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, Scalar<V> inc)
    {
        changeEval(f, x, [&](auto...){ return inc; });
    }

    template <class F, class V, class StepSize, std::enable_if_t<!std::is_fundamental<StepSize>::value, int> = 0>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, const StepSize& stepSize)
    {
        changeEval(f, x, stepSize, handy::range(x.size()));
    }

    template <class F, class V, typename Int>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, Scalar<V> inc, handy::Range<Int> range)
    {
        changeEval(f, x, [&](auto...){ return inc; }, range);
    }

    template <class F, class V, class StepSize, typename Int, std::enable_if_t<!std::is_fundamental<StepSize>::value, int> = 0>
    static void changeEval (F f, const Eigen::MatrixBase<V>& x, const StepSize& stepSize, handy::Range<Int> range)
    {
        Eigen::Matrix<Scalar<V>, V::RowsAtCompileTime, V::ColsAtCompileTime> y = x;

        for(auto i : range)
        {
            auto inc = stepSize(x, i);

            y(i) = x(i) - inc;

            f(y, i, inc);

            y(i) = x(i);
        }
    }
    //@}
};







/** @brief Central difference class
 * 
 *  @details This is the actual implementation of the gradients and hessian finite differences for central
 *           difference. Calls are delegated bie CRTP
 * 
 *  @tparam Function Functor type
 *  @tparam Step Step size template functor
 *  @tparam Float Base floating point type
*/
template <class Function, class Step>
struct Central : public FiniteDifference<Central<Function, Step>>
{
    CPPOPT_USING_FINITE_DIFFERENCE(Base, FiniteDifference<Central<Function, Step>>);

    
    /** @name
     *  @brief Gradient calculation for central difference
     * 
     *  @details Calculate the gradient of @c f at a given vector or scalar @c x, given a already calculated
     *           (via base CRTP class) function value @c fx = @c f(x). Also, given an direction @c e, calculate
     *           the directional derivative in the direction of @c e.
     * 
     *  @param x The scalar or Eigen::MatrixBase value where the gradient (or first derivative, if scalar) of @c f will be evaluated
     *  @param fx The result of @c f applied to @c x, that is, f(x)
     *  @param g A reference to where we will write the gradient of @c f calculated at @c x
     *  @param e The direction to calculate the directional derivative of @f c at @c x in the direction of @c e
     * 
     *  @note Because we do not use the value of @c fx (@c f(x)) for central difference calculations, all calls to the 
     *        gradient will have an overload without @c fx. The overload with @c fx will simply delegate the call 
    */
    //@{
    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x, Scalar<V>) const
    {
        return gradient(x);
    }

    template <class V>
    auto gradient (const Eigen::MatrixBase<V>& x) const
    {
        Plain<V> g(x.rows(), x.cols());

        gradient(x, g);

        return g;
    }


    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g, Scalar<V>) const
    {
        gradient(x, g);
    }

    template <class V>
    void gradient (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        Plain<V> y = x;

        for(int i = 0; i < x.size(); ++i)
        {
            auto h = step(x, i);

            y(i) = x(i) - h;

            auto fl = f(y);

            y(i) = x(i) + h;

            g(i) = (f(y) - fl) / (2 * h);

            y(i) = x(i);
        }
    }

    template <class V, class U>
    auto gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V>) const
    {
        return gradientDir(x, e);
    }

    template <class V, class U>
    auto gradient (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return gradientDir(x, e);
    }
    //@}



    /** @name
     *  @brief Hessian calculation with hessian difference
     * 
     *  @details Calculate the hessian of @c f at a given vector or scalar @c x, given a already calculated
     *           (via base CRTP class) function value @c fx = @c f(x). Also, given an direction @c e, calculate
     *           the hessian product: @f$\nabla^2 f(x) e$@f.
     * 
     *  @param x The scalar or Eigen::MatrixBase value where the hessian (or second difference, if scalar) of @c f will be evaluated
     *  @param fx The result of @c f applied to @c x, that is, f(x). If it is a vector, we return the Jacobian of @c f.
     *  @param e The direction to calculate the hessian vector product @f$\nabla^2 f(x) e$@f.
    */
    //@{
    template <class V>
    void hessian (const Eigen::MatrixBase<V>& x, Plain2D<V>& hess, Scalar<V> fx) const
    {
        using Float = Scalar<V>;

        Plain<V> y = x;

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
    }

    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, Scalar<V> fx) const
    {
        Plain2D<V> hess(x.size(), x.size());
        hessian(x, hess, fx);

        return hess;
    }

    template <class V, class U, class W>
    auto hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>&) const
    {
        return hessianDir(x, e);
    }

    template <class V, class U, class W>
    auto hessian (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        return hessianDir(x, e);
    }
    //@}



    /** @name
     *  @brief Central finite approximation of the gradient of @c f
     * 
     *  @details This is a very general routine that can be used for both vector and scalar inputs.
     * 
     *  @param x A scalar or Eigen::MatrixBase matrix/vector
     *  @param e A scalar or Eigen::MatrixBase matrix/vector
     *  @param fx A scalar or Eigen::MatrixBase matrix/vector resulting from a call to @c f(x)
     *  @param h The infinitezimal step size
     *  @returns f(x)
    */
    //@{
    template <class V, class U>
    auto gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, Scalar<V>) const
    {
        return gradientDir(x, e);
    }

    template <class V, class U>
    auto gradientDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        auto h = step(x);
        return (f(x + h * e) - f(x - h * e)) / (2 * h);
    }

    template <class V, class U, class W>
    auto hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e, const Eigen::MatrixBase<W>&) const
    {
        return hessianDir(x, e);
    }

    template <class V, class U, class W>
    auto hessianDir (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<U>& e) const
    {
        auto h = step(x);
        return (gradient(x + h * e) - gradient(x - h * e)) / (2 * h);
    }
    //@}
};


/** @name
 *  @brief Classes used to join everything and expose interface via @c operator().
 *  
 *  @tparam Function The functor we are going to work with
 *  @tparam Difference The template describing which finite difference we will use
 *  @tparam Step The step size template
 *  @tparam Float Base floating point type
*/
//@{

/// Gradient interface for finite difference estimation
template <class Function, template <class, class> class Difference = Forward, class Step = AutoStep>
struct Gradient : public Difference<wrap::Functions<wrap::Conditions::Function, Function>, Step>
{
    using Base = Difference<wrap::Functions<wrap::Conditions::Function, Function>, Step>;
    using Base::Base;
    using Base::gradient;

    // Hide base implementation
    void hessian () {}


    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        return gradient(x);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx) const
    {
        return gradient(x, fx);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g) const
    {
        return gradient(x, g);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, Plain<V>& g, ::nlpp::impl::Scalar<V> fx) const
    {
        return gradient(x, g, fx);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e) const
    {
        return gradient(x, e);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, const Eigen::MatrixBase<V>& e, ::nlpp::impl::Scalar<V> fx) const
    {
        return gradient(x, e, fx);
    }
};


/// Hessian interface for finite difference estimation
template <class Function, template <class, class> class Difference = Forward, class Step = SimpleStep<>, typename Float = types::Float>
struct Hessian : public Difference<wrap::Functions<wrap::Conditions::Function, Function>, Step>
{
    using Base = Difference<wrap::Functions<wrap::Conditions::Function, Function>, Step>;
    using Base::hessian;

    /** @name
     *  @brief Default constructor must have @c h equal to @f$u^{\frac{3}{4}}$@f
    */
    //@{
    Hessian (const Function& f) : Hessian(f, std::pow(constants::eps_<Float>, 1.0 / 2)) {}

    Hessian (const Function& f, const Step& step) : Base(f, step) {}
    //@}

    // Hide base implementation
    void gradient () {}


    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, Plain2D<V>& h) const
    {
        return hessian(x, h);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, Plain2D<V>& h, ::nlpp::impl::Scalar<V> fx) const
    {
        return hessian(x, h, fx);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        return hessian(x);
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, ::nlpp::impl::Scalar<V> fx) const
    {
        return hessian(x, fx);
    }
};
//@}



/** @name
 *  @brief Some default step size functors
 *  
 *  @details These functors return a scalar @c h that represents the infinitezimal step size used in the finite
 *           difference calculations. Given a scalar or vector @c x and a position @c i, the interface must have:
 * 
 *           - A default constructor
 *
 *           - An <tt> void init(const X&) </tt>, that will gather some information before the calls to <tt> operator(const X&, int) </tt>
 *           - An @c Float operator()(const X&), that will return a default value for the step size to be taken at @c x
 *           - An @c Float operator(const X&, int), that will return a default value for the step size to be taken at @f$x_i$@f
 * 
 *  @tparam Float The base floating point type
*/
struct AutoStep
{
    AutoStep (...) {}

    template <class V>
    impl::Scalar<V> operator () (const Eigen::MatrixBase<V>&, ...) const
    {
        return constants::eps_<impl::Scalar<V>>;
    }
};


template <typename Float>
struct SimpleStep
{
    SimpleStep (Float h = constants::eps_<Float>) : h(h) {}

    Float operator () (...) const
    {
        return h;
    }
 
    Float h;
};


template <typename Float>
struct NormalizedStep : public SimpleStep<Float>
{
    CPPOPT_USING_STEPSIZE(Base, SimpleStep<Float>);

    /// Returns @f$max(u^2, |x_i|)$@f for dimension @c i
    template <class V>
    Float operator () (const Eigen::MatrixBase<V>& x, int i) const
    {
        Float step = h * std::abs(x[i]);

        if(step <= minValue)
            step = constants::eps_<Float>;

        return step;
    }

    static constexpr Float minValue = constants::eps_<Float> * constants::eps_<Float>;
};
//@}


/** @name
 *  @brief Simple functions used to delegate the call to the given classes
 * 
 *  @tparam @tparam Difference The template describing which finite difference we will use
 *  @tparam Function The functor we are going to work with
 *  @tparam Step The step size template
 *  @tparam Float Base floating point type
*/
//@{
template <class Function, template <class, class> class Difference = Forward, class Step = AutoStep>
auto gradient (const Function& f)
{
    return Gradient<Function, Difference, Step>(f);
}

template <class Function, template <class, class> class Difference = Forward, class Step = AutoStep>
auto gradient (const Function& f, const Step& step)
{
    return Gradient<Function, Difference, Step>(f, step);
}


template <class Function, template <class, class> class Difference = Forward, class Step = SimpleStep<>>
auto hessian (const Function& f)
{
    return Hessian<Function, Difference, Step>(f);
}

template <class Function, template <class, class> class Difference = Forward, class Step = AutoStep>
auto hessian (const Function& f, const Step& step)
{
    return Hessian<Function, Difference, Step>(f, step);
}
//@}

//@}

} // namespace fd

} // namespace nlpp