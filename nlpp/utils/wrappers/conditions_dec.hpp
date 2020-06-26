#pragma once

#include "helpers/helpers_dec.hpp"

namespace nlpp::wrap
{

enum Conditions : std::size_t
{
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

} // namespace nlpp::wrap