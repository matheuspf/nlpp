/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include "include.hpp"
#include "types.hpp"

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



#define NLPP_USING_POLY_CLASS(ClassName, BaseName, ...) \
	using BaseName = __VA_ARGS__;	\
	using BaseName::BaseName;		\
	using BaseName::set;	 		\
	using BaseName::impl;			\
									\
	template <typename... Args>		\
	ClassName(Args&&...args) : BaseName(std::forward<Args>(args)...) {}


#define NLPP_FUNCTION_TRAITS(NAME, FUNCTION) \
template <class Cls, class... Args> \
struct NLPP_CONCAT(NAME, Traits) \
{ \
    using ReturnType = std::conditional_t<impl(nullptr), decltype(std::declval<Cls>().FUNCTION(std::declval<Args>()...)), std::nullptr_t>; \
 \
    enum { Has = impl(nullptr) } \
 \
    static constexpr bool impl (decltype(std::declval<Cls>().FUNCTION(std::declval<Args>()...), void())*) \
    { \
        return true; \
    } \
 \
    static constexpr bool impl (...) \
    { \
        return false; \
    } \
}; \
\
template <class Cls, class... Args> \
using NLPP_CONCAT(NAME, ReturnType) = typedef NLPP_CONCAT(NAME, Traits)::ReturnType; \
\
template <class Cls, class... Args> \
constexpr bool NLPP_CONCAT(Has, NAME) = NLPP_CONCAT(NAME, Traits)::Has;



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


namespace wrap
{

/** @name
 *  @brief Decides whether a given function has or has not a overloaded member functions taking the given parameters
*/
//@{
HAS_OVERLOADED_FUNC(operator(), HasOperator);

HAS_OVERLOADED_FUNC(function, HasFunction);

HAS_OVERLOADED_FUNC(gradient, HasGradient);
//@}

} // namespace wrap


namespace impl
{

/// A functor that does nothing
struct NullFunctor
{
	void operator () (...) {}
};


template <typename>
struct PrintType;

/** @name
 *  @brief Tells if @c T is an Eigen::EigenBase (a vector/matrix) or an scalar (a float or int)
*/
//@{
template <typename T>
struct IsMat
{
	template <class U>
	static constexpr bool impl (Eigen::EigenBase<U>*) { return true; }

	static constexpr bool impl (...) { return false; }


	enum { value = impl((std::decay_t<T>*)0) };
};

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
constexpr bool isScalar = IsScalar<T>::value;
//@}


template <typename T, class V>
constexpr decltype(auto) cast (V&& v);

template <class V>
std::string toString (const V& x);

/** @name
 *  @brief Define function overloading calling precedence
*/
//@{
template <int I = 0, int Max = 10>
struct Precedence : Precedence<I+1, Max> {};

template <int I>
struct Precedence <I, I> {};
//@}


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
inline constexpr decltype(auto) shift (T&& x);

template <typename T, typename U, typename... Args>
inline constexpr decltype(auto) shift (T&& x, U&& y, Args&&... args);
//@}

/// Implement Matlab's sign function
template <typename T>
inline constexpr int sign (T t);

template <class...> constexpr std::false_type always_false{};

} // namespace nlpp
