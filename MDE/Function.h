/** \file Function.h
  *
  * Defines classes for wrapping variables and functions needed by MDE.
*/

#ifndef MDE_FUNCTION_BASE_H
#define MDE_FUNCTION_BASE_H

#include <algorithm>
#include <cmath>
#include <numeric>
  
#include "Vector.h"


namespace mde
{

/** Base 'mde::Function'. All user defined functions useed by MDE should inherit 
  * from this class. Here we defined the number of variables, the 'Vector' type
  * the lower and upper bounds, the optimal value for the function and dummy
  * functions for the equalities and inequalities values. Every parameter has a 
  * default value. For example, you may not specify the lower and upper bounds, but 
  * must specify the number of variables 'N'. Otherwise, if you set either lower or 
  * upper bounds, you don't need to inform the number of variables. The 'optimal' 
  * value is also optional. If the function drops bellow this value, MDE stops immediately.
*/
template<int NumVariables = 0>
struct Function
{
    /// Vector type. If 'NumVariables' is given, it is a 'std::array'. Otherwise, it is a 'std::vector'
    using Vector = help::Vector<NumVariables>;


    /// Dummy value for the optimal parameter.
    Function (double optimal = -1e8) : optimal(optimal), lowerBounds(NumVariables, -1e8),
                                       upperBounds(NumVariables, 1e8) {}


    /// The user may or may not supply the number of variables 'N'
    Function (int N, double optimal = -1e8) : N(N), lowerBounds(N, -1e8), upperBounds(N, 1e8), 
                                              optimal(optimal) {}


    /// In the case of 'NumVariables' is given
    Function (double l, double u, double optimal = -1e8) : Function(NumVariables, l, u, optimal) {}


    /// In the case it is not, but a runtime value is
    Function (int N, double l, double u, double optimal = -1e8) : N(N), lowerBounds(N), upperBounds(N), 
                                                                  optimal(optimal)
    {
        std::fill(lowerBounds.begin(), lowerBounds.end(), l);
        std::fill(upperBounds.begin(), upperBounds.end(), u);
    }


    /// For passing lower and upper bounds right in the constructor
    Function (Vector lowerBounds, Vector upperBounds, double optimal = -1e8) : N(NumVariables), 
                                                                               lowerBounds(lowerBounds),
                                                                               upperBounds(upperBounds),
              
                                                                              optimal(optimal) {}
    /// Here with the number of variables at runtime
    Function (int N, Vector lowerBounds, Vector upperBounds, double optimal = -1e8) : N(N), lowerBounds(N),
                                                                                      upperBounds(N), 
                                                                                      optimal(optimal) {}


    /** These are dummy constraint functions. You can override them in your class or not.
      * It is optional. Notice here that we are returning a 'double'. You can also define
      * these functions to return any iterable type (that is, has both 'std::begin' and 
      * 'std::end' defined) containing the value for each equality/inequality. The 
      * 'SetValues' class will handle them.
    */
    double equalities (const Vector&) const
    {
        return 0.0;
    }

    double inequalities (const Vector&) const
    {
        return 0.0;
    }


    /** If the number of variables is given by 'NumVariables', it is initalized here.
      * If you supply an 'N' either on your function or here, this value will be overrided.
      * If the number of variables is not given in anyway, you must give a proper value to
      * the lower or upper bounds.
    */
    int N = NumVariables;


    /// Optimal value. Only used for convergence. Totally optional.
    double optimal;


    /// Lower and upper bounds. Defaulted to '-1e8' and '1e8' to avoid loss of precision.
    Vector lowerBounds;
    Vector upperBounds;
};




/** The above class is meant for inheritance on the user defined function. This one
  * inherits from the user function, adding the interface used by MDE. It defines
  * a single function, the 'operator()', that takes a 'Vector' as reference and sets
  * both its fitness and violation, based on the functions given by the user.
*/

template<class Func>
class SetValues : public Func
{
public:

    /// 'Vector' type, from the 'mde::Function' class
    using Vector = typename Func::Vector;


    /** The constructor delegates the initialization of the base class. The parameter
      * 'eqTol' is a tolerance factor for the equality constraints.
    */
    SetValues (double eqTol = 1e-10, const Func& func = Func()) : eqTol(eqTol), Func(func) {}



    /** The only function available. It simply delegates the call to the function
      * 'operator()' returning the function value given returning a 'double', which 
      * must be defined by the user. The 'violation' variable is set to the sum of
      * the violation of each equality and inequality.
    */
    double operator () (Vector& x)
    {
        x.fitness = Func::operator()(x);

        x.violation = inequalitiesValue(x) + equalitiesValue(x);

        return x.fitness;
    }



private:


    /** If the function provided by the user has equalities or inequalities constraints,
      * you can define them to return either a float or double value, or an iterable
      * container (with 'std::begin' and 'std::end' defined) containing the values of
      * each equality/inequality.
      *
      * All inequalities are of the form:
      *
      *  g(x) <= 0.0
      *
      * The penalty is therefore:
      *
      * max(0.0, abs(g(x)))
    */
    template <class F = Func, std::enable_if_t<std::is_floating_point<
              decltype(std::declval<F>().inequalities(Vector{}))>::value, int> = 0>
    double inequalitiesValue (const Vector& x)
    {
        return std::max(0.0, Func::inequalities(x));
    }



    /** All equalities are transformed into inequalities of the form:
      *
      * abs(h(x)) - eqTol <= 0
      *
      * And the penalty:
      *
      * 0.0         if abs(h(x)) <= 0
      * abs(h(x))   otherwise
      *
      * In general, 'eqTol' is a very small value (set by default to 1e-10), which
      * avoids floating point precision issues. Anyways, if you really want an
      * equality (although the function may never converge), simply set 'eqTol' to 0.
    */
    template <class F = Func, std::enable_if_t<std::is_floating_point<
              decltype(std::declval<F>().equalities(Vector{}))>::value, int> = 0>
    double equalitiesValue (const Vector& x)
    {
        double eqVal = Func::equalities(x);

        return std::abs(eqVal) < eqTol ? 0.0 : std::abs(eqVal);
    }




    /// Same as above, but applied to every element of the container returned by 'Func::inequalities'
    template <class F = Func, std::enable_if_t<!std::is_floating_point<
              decltype(std::declval<F>().inequalities(Vector{}))>::value, int> = 0>
    double inequalitiesValue (const Vector& x)
    {
        const auto& ineqs = Func::inequalities(x);

        return std::accumulate(ineqs.begin(), ineqs.end(), 0.0, [this](double sum, double x)
        {
            return sum + std::max(0.0, x);
        });
    }


    /// Same as above, but applied to every element of the container returned by 'Func::equalities'
    template <class F = Func, std::enable_if_t<!std::is_floating_point<
              decltype(std::declval<F>().equalities(Vector{}))>::value, int> = 0>
    double equalitiesValue (const Vector& x)
    {
        const auto& eqs = Func::equalities(x);

        return std::accumulate(eqs.begin(), eqs.end(), 0.0, [this](double sum, double x)
        {
            return sum + (std::abs(x) < eqTol ? 0.0 : std::abs(x));
        });
    }


    /// Tolerance parameter for equality constraints
    double eqTol;
};

} // namespace mde



#endif //MDE_FUNCTION_BASE_H
