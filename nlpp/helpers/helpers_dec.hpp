/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include "include.hpp"
#include "types.hpp"
#include "traits.hpp"
#include "forward_declarations.hpp"

/// Expands variadic arguments
#define NLPP_EXPAND(...) __VA_ARGS__

/// Concatenate two tokens
#define NLPP_CONCAT(x, y) NLPP_CONCAT_(x, y)

/// @copybrief CONCAT
#define NLPP_CONCAT_(x, y) NLPP_EXPAND(x ## y)
//@}

//@{
/** @brief Count number of variadic arguments
*/
#define NLPP_NUM_ARGS_(_1, _2 ,_3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

/// @copybrief NUM_ARGS_
#define NLPP_NUM_ARGS(...) NLPP_NUM_ARGS_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
//@}

/// This guy will call MACRON, where @c N is the number of variadic arguments
#define NLPP_APPLY_N(MACRO, ...) NLPP_EXPAND(NLPP_CONCAT(MACRO, NLPP_NUM_ARGS(__VA_ARGS__)))(__VA_ARGS__)
//@}

#define NLPP_ENUM_OPERATOR(CLASS, OP, INT) \
inline constexpr CLASS operator OP (CLASS a, CLASS b)   \
{   \
    return static_cast<CLASS>(static_cast<INT>(a) OP static_cast<INT>(b));  \
}


#define NLPP_USING_POLY_CLASS(ClassName, BaseName, ...) \
	using BaseName = __VA_ARGS__;	\
	using BaseName::BaseName;		\
	using BaseName::set;	 		\
	using BaseName::impl;			\
									\
	template <typename... Args>		\
	ClassName(Args&&...args) : BaseName(std::forward<Args>(args)...) {}


#define NLPP_HAS_MEMBER(NAME, OP) \
template <class T, class... Args> \
using NAME = decltype(std::declval<T>().OP(std::declval<Args>()...));


namespace nlpp
{

namespace poly
{

template <class Impl>
struct CloneBase
{
    std::unique_ptr<Impl> clone () const;
    virtual Impl* clone_impl () const = 0;
};


template <class Base>
struct PolyClass
{
    virtual ~PolyClass () = default;

    PolyClass (std::unique_ptr<Base> ptr);
    PolyClass (const PolyClass& polyClass);
    PolyClass (PolyClass&& polyClass) = default;

    template <class Derived, std::enable_if_t<std::is_base_of_v<Base, Derived>, int> = 0>
    PolyClass (Derived&& derived);


    PolyClass& operator= (PolyClass&& polyClass) = default;

    PolyClass& operator= (std::unique_ptr<Base> ptr);

    template <class Derived, std::enable_if_t<std::is_base_of_v<Base, Derived>, int>>
    PolyClass& operator= (Derived&& derived);

    PolyClass& operator= (const PolyClass& polyClass);

    Base* get () const;

    Base* operator-> () const;

	template <class T>
	void set (T&& t);

    std::unique_ptr<Base> impl;
};

} // namespace poly


namespace impl
{

/** @name
 *  @brief Tells if @c T is an Eigen::EigenBase (a vector/matrix) or an scalar (a float or int)
*/
//@{
template <typename T>
struct IsEigen
{
    template <class U>
    static constexpr int impl (Eigen::EigenBase<U>*)
    {
        return Plain<U>::ColsAtCompileTime != 1 ? 1 : 2;
    }

    static constexpr int impl (...) { return 0; }

    enum { value = impl((std::decay_t<T>*)0) };
    enum { isMat = IsEigen::value == 1, isVec = IsEigen::value == 2 };
};


template <typename T>
struct IsMat : public std::bool_constant<bool(IsEigen<T>::isMat)> {};

template <typename T>
struct IsVec : public std::bool_constant<bool(IsEigen<T>::isVec)> {};


template <typename T>
struct IsScalar
{
	enum
	{
		value = (std::is_floating_point<std::decay_t<T>>::value || std::is_integral<std::decay_t<T>>::value)
	};
};

template <typename T>
constexpr bool isMat = IsMat<T>::value;

template <typename T>
constexpr bool isVec = IsVec<T>::value;

template <typename T>
constexpr bool isScalar = IsScalar<T>::value;
//@}


template <class V>
using Plain2D = std::conditional_t<isMat<V>, Plain<V>,  Eigen::Matrix<Scalar<V>, V::RowsAtCompileTime, V::RowsAtCompileTime>>;

/** @name
 *  @brief Define function overloading calling precedence
*/
//@{
template <int I = 0, int Max = 10>
struct Precedence : Precedence<I+1, Max> {};

template <int I>
struct Precedence <I, I> {};
//@}


/** @name
 *  @brief Implementation of the is_detected helper available in std::experimental
 *         and described in https://en.cppreference.com/w/cpp/experimental/is_detected
*/
//@{
struct nonesuch
{
    //~nonesuch() = delete;
    //nonesuch(nonesuch const&) = delete;
    //void operator=(nonesuch const&) = delete;
};

template <class Default, class AlwaysVoid, template<class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type = Default;
};
 
template <class Default, template<class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type = Op<Args...>;
};
 
template <template<class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <template<class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;
 
template <template<class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;
 
template <class Default, template<class...> class Op, class... Args>
using detected_or = detector<Default, void, Op, Args...>;

//@}

template <class...>
constexpr std::false_type always_false{};

template <typename T, class V>
constexpr decltype(auto) cast (V&& v)
{
    return v.template cast<T>();
}

template <bool Ok, std::size_t I, typename... Args>
struct NthArgImpl_2
{
    using type = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <std::size_t I, typename... Args>
struct NthArgImpl_2<false, I, Args...>
{
    using type = nonesuch;
};


template <std::size_t I, typename... Args>
struct NthArgImpl : public NthArgImpl<I, std::tuple<Args...>>
{
};

template <std::size_t I, typename F, typename S>
struct NthArgImpl<I, std::pair<F, S>> : public NthArgImpl<I, std::tuple<F, S>>
{
};

template <std::size_t I, typename... Args>
struct NthArgImpl<I, std::tuple<Args...>> : public NthArgImpl_2<(I >=0 && I < sizeof...(Args)), I, Args...>
{
};

template <std::size_t I, typename... Args>
using NthArg = typename NthArgImpl<I, Args...>::type;

template <typename... Args>
using FirstArg = NthArg<0, Args...>;

template <typename> struct PrintType;

template <typename...> struct EmptyBase
{
    EmptyBase(...) {}
};

} // namespace impl



/// Definition of some constants
namespace constants
{

template <typename T = types::Float>
constexpr T pi_ = T(3.14159265359);

template <typename T = types::Float>
constexpr T phi_ = T(1.61803398875);

template <typename> struct Eps;

// TODO: automatically select the sqrt of the std::numeric_limits::epsilon of each type
constexpr long double eps_f (long double) { return 3e-10; }
constexpr double eps_f (double) { return 1e-8; }
constexpr float eps_f (float) { return 1e-4; }

template <typename T = types::Float>
constexpr T eps_ = eps_f(T{});

constexpr double eps = eps_<types::Float>;
constexpr double pi  = pi_<types::Float>;
constexpr double phi = phi_<types::Float>;

} // namespace constants

/** @name
 *  @brief Useful for some line search operations
*/
//@{
template <typename T>
inline constexpr decltype(auto) shift (T&& x)
{
    return std::forward<T>(x);
}

template <typename T, typename U, typename... Args>
inline constexpr decltype(auto) shift (T&& x, U&& y, Args&&... args)
{
    x = std::forward<U>(y);

    return shift(std::forward<U>(y), std::forward<Args>(args)...);
}
//@}

/// Implement Matlab's sign function
template <typename T>
inline constexpr int sign (T t)
{
    return int(T{0} < t) - int(t < T{0});
}

} // namespace nlpp