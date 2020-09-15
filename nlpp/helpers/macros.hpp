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
