#pragma once

#include "status_dec.hpp"


namespace nlpp
{

namespace
{

bool Status::ok () const
{
    return code == Code::Ok;
}

bool Status::error () const
{
    return code >= Code::NotOk;
}

Status::operator bool() const
{
    return ok();
}

std::string Status::toString() const
{
    // return std::to_string(std::size_t(code));

    std::string codeString;

    auto addCode = [&codeString](const std::string& c)
    {
        codeString += (codeString.size() ? " | " : "") + c;
    };

    if(code == Code::Ok)
        return "OK";

    if(bool(code & Code::VariableCondition))
        addCode("VariableCondition");

    if(bool(code & Code::FunctionCondition))
        addCode("FunctionCondition");

    if(bool(code & Code::GradientCondition))
        addCode("GradientCondition");

    if(bool(code & Code::HessianCondition))
        addCode("HessianCondition");

    return codeString;
}

void Status::set (Status::Code newCode)
{
    code = newCode;
}


std::ostream& operator<< (std::ostream& out, const Status& status)
{
    return out << status.toString();
}

bool operator == (const Status& status, const Status::Code& code)
{
    return status.code == code;
}

bool operator == (const Status::Code& code, const Status& status)
{
    return code == status.code;
}

} // namespace

}