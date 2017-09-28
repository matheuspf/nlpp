/** \file Vector.h
  *
  * General 'Vector' type containing information about fitness and
  * constraints values. If the size is specified at compile time, the
  * class will inherit from 'std::array'. Otherwise, it will inherit
  * from 'std::vector'. Also, if the size specified at compile time is 
  * too big, we use a 'std::vector' to avoid stack overflow. Some
  * construtor definitions were made to make the interface of
  * 'std::array' compatible.
  *
*/


#ifndef MDE_VECTOR_H
#define MDE_VECTOR_H

#include <type_traits>
#include <vector>
#include <array>

namespace mde
{

namespace help
{

/// Selects 'std::array' if '(N == 0 || N >= maxSize * sizeof(T)'. Else, selects 'std::vector'
template<typename T, std::size_t N>
struct SelectType
{
    /// Maximum size to allow the creation of 'Vector' on the stack.
    static constexpr std::size_t maxSize = 10000 * sizeof(char);

    using type = std::conditional_t<(N == 0 || N >= maxSize * sizeof(T)), std::vector<T>, std::array<T, N>>;
};

/// Alias
template<typename T, std::size_t N>
using SelectType_t = typename SelectType<T, N>::type;



/// 'VectorBase' class, inheriting from the selected type.
template<typename T, std::size_t N = 0>
struct VectorBase : public SelectType_t<T, N>
{
    /// Inherits all constructors of std::vector, given that std::array has no constructor
    using Base = SelectType_t<T, N>;
    using Base::Base;


    /// Just for convenience
    static constexpr bool isArray =  std::is_same<std::array<T, N>, Base>::value;
    static constexpr bool isVector = std::is_same<std::vector<T>, Base>::value;



    VectorBase() : Base() {}

    /// Constructor needed for list initialization if inheriting from std::array
    template<typename U = Base, typename = std::enable_if_t<std::is_same<std::array<T, N>, U>::value>>
    VectorBase(std::initializer_list<T> in)
    {
        std::copy(in.begin(), in.end(), this->begin());
    }

    /// Constructor for std::array to make the interface uniform with std::vector
    template<typename U = Base, typename = std::enable_if_t<std::is_same<std::array<T, N>, U>::value>>
    VectorBase(int, const T& t = T{})
    {
        std::fill(this->begin(), this->end(), t);
    }
};




/** The actual 'Vector' class. Uses 'double' and stores both
  * fitness and violation values, initially set to a very large
  * value (1e18), so that this vector is worst than anything.
*/
template<std::size_t N = 0>
struct Vector : public help::VectorBase<double, N>
{
    /// Inherits all the constructors
    using help::VectorBase<double, N>::VectorBase;


    /// Feasible condition: no violation
    inline bool feasible () const { return violation == 0.0; }


    /** The comparison function used in MDE. If both vectors are 
      * feasible (violation == 0.0), the one with smaller fitness 
      * is better. If onevector is feasible and the other is not, 
      * the feasible vector is better. If both vectors are infeasible, 
      * the one with smallest violation is better.
    */
    bool operator < (const Vector& x) const
    {
        if(this->feasible() && x.feasible())
            return this->fitness < x.fitness;

        if(this->feasible())
            return true;

        if(x.feasible())
            return false;

        return this->violation < x.violation;
    }
    

    /// Fitness and violation values
    double fitness = 1e18;
    double violation = 1e18;
};


} // namespace help

} // namespace mde



#endif //MDE_VECTOR_H
