#pragma once

#include "helpers/helpers_dec.hpp"

namespace nlpp::wrap
{

enum class Conditions : std::size_t
{
    Empty                       = 0,

    Function                    = 1 << 0,
    Gradient                    = 1 << 1,
    Hessian                     = 1 << 2,
    AllFunctions                = (1 << 3) - 1,

    Start                       = 1 << 10,
    Bounds                      = 1 << 11,
    LinearEqualities            = 1 << 12,
    LinearInequalities          = 1 << 13,
    FullDomain                  = ((1 << 14) - 1) & ~((1 << 10) - 1),

    NLEqualities                = 1 << 20,
    NLInequalities              = 1 << 21,
    NLEqualitiesJacobian        = 1 << 22,
    NLInequalitiesJacobian      = 1 << 23,
    FullNL                      = ((1 << 24) - 1) & ~((1 << 20) - 1)
};

NLPP_ENUM_OPERATOR(Conditions, |, std::size_t)
NLPP_ENUM_OPERATOR(Conditions, &, std::size_t)


namespace impl
{

using ::nlpp::impl::Scalar, ::nlpp::impl::Plain, ::nlpp::impl::Plain2D,
      ::nlpp::impl::isMat, ::nlpp::impl::isVec, ::nlpp::impl::detected_t,
      ::nlpp::impl::is_detected_v, ::nlpp::impl::always_false, ::nlpp::impl::NthArg;


NLPP_MAKE_CALLER(CallOp, true);


// template <template <class, class> class Check, class V, class TFs, class Idx>
// struct OpIdImpl;

// template <template <class, class> class Check, class V, class... Fs, std::size_t... Is>
// struct OpIdImpl<Check, V, std::tuple<Fs...>, std::index_sequence<Is...>>
// {
//     enum { value = (int(Check<Fs, V>::value * int(Is + 1)) + ...) - 1 };
// };

// template <template <class, class> class Check, class V, class TFs>
// static constexpr int OpId = OpIdImpl<Check, V, TFs, std::make_index_sequence<std::tuple_size_v<TFs>>>::value;


template <template <class, class> class Check, class V, class TFs, std::size_t I, std::size_t IMax>
struct OpIdImpl : std::integral_constant<int, Check<std::tuple_element_t<I, TFs>, V>::value ? int(I) :
                                              OpIdImpl<Check, V, TFs, I+1, IMax>::value > {};

template <template <class, class> class Check, class V, class TFs, std::size_t IMax>
struct OpIdImpl<Check, V, TFs, IMax, IMax> : std::integral_constant<int, -1> {};


template <template <class, class> class Check, class V, class TFs>
static constexpr int OpId = OpIdImpl<Check, V, TFs, 0, std::tuple_size_v<TFs>>::value;


template <template <class, class> class Check, class V, class TFs>
static constexpr bool HasOp = OpId<Check, V, TFs> >= 0;

template <template <class, class> class Check, class V, class TFs, std::size_t I>
static constexpr bool HasOpId = (I < std::tuple_size_v<TFs>) && (HasOp<Check, V, std::tuple<std::tuple_element_t<std::min(I, std::tuple_size_v<TFs>-1), TFs>>> >= 0);



template <class T, class... Args>
using OperatorType = detected_t<std::invoke_result_t, T, Args...>;


template <template <class...> class Check, class TFs, class V, std::size_t... Is>
constexpr int opIdImpl (std::index_sequence<Is...>)
{
    std::initializer_list<bool> arr = { Check<std::tuple_element_t<Is, TFs>, V>::value... };

    std::size_t id = std::max_element(arr.begin(), arr.end()) - arr.begin();

    if(id == 0 && *arr.begin() == 0)
        return -1;
    
    return id;
}

template <template <class...> class Check, class TFs, class V = std::nullptr_t>
constexpr int opId = opIdImpl<Check, TFs, V>(std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TFs>>>{});

template <template <class...> class Check, class TFs, class V = std::nullptr_t>
constexpr bool hasOp = opId<Check, TFs, V> >= 0;


} // namespace impl

} // namespace nlpp::wrap