#pragma once

#include "macros.hpp"

namespace nlpp
{

// Since status is not a template
namespace
{

struct Status
{
    enum class Code : std::size_t
    {
        Ok                          = 0,
        NumIterations               = 1 << 0,
        VariableCondition           = 1 << 1,
        FunctionCondition           = 1 << 2,
        GradientCondition           = 1 << 3,
        HessianCondition            = 1 << 4,

        NotOk                       = 1 << 10,
        FunctionNAN                 = 1 << 11,
        GradientNAN                 = 1 << 12,
        UnknowError                 = 1 << 13,
    };

    Status(Code code = Code::Ok) : code(code) {}

    bool ok() const;
    bool error() const;
    operator bool() const;

    std::string toString() const;

    void set (Status::Code newCode);

    Code code;
};

std::ostream& operator<< (std::ostream& out, const Status& status);
// std::ostream& operator<< (std::ostream& out, const Status::Code& code);
bool operator == (const Status&, const Status::Code&);
bool operator == (const Status::Code&, const Status&);

NLPP_ENUM_OPERATOR(Status::Code, |, std::size_t)
NLPP_ENUM_OPERATOR(Status::Code, &, std::size_t)

} // namespace


} // namespace nlpp