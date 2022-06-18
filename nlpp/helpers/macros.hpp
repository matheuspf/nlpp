#pragma once

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

#define NLPP_MAKE_CALLER(NAME, HAS_CALL_OP) \
\
template <class T, class... Args> \
using NLPP_CONCAT(NAME, Invoke) = decltype(std::declval<T>().NAME(std::declval<Args>()...));   \
\
template <class Impl, class... Args> \
auto NLPP_CONCAT(NAME, Call) (const Impl& impl, Args&&... args) \
{ \
    if constexpr(::nlpp::impl::is_detected_v<NLPP_CONCAT(NAME, Invoke), Impl, Args...>) \
        return impl.NAME(std::forward<Args>(args)...); \
\
    else if constexpr(HAS_CALL_OP &&::nlpp::impl::is_detected_v<std::invoke_result_t, Impl, Args...>) \
        return impl(std::forward<Args>(args)...); \
\
    else \
        return ::nlpp::impl::nonesuch{}; \
} \
\
template <class Impl, class... Args> \
using NLPP_CONCAT(NAME, Type) = decltype(NLPP_CONCAT(NAME, Call)(std::declval<Impl>(), std::declval<Args>()...));



#define NLPP_MAKE_CALLER__(NAME, HAS_CALL_OP) \
\
template <class F, class... Args> \
concept NLPP_CONCAT(NAME, Functor) = \
requires(const F& f, Args&&... args) { { f.NAME(std::forward<Args>(args)...) } } || \
(HAS_CALL_OP && requires(const F& f, Args&&... args) { { f(std::forward<Args>(args)...) } }); || \
