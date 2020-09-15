#pragma once

#include "status_dec.hpp"


namespace nlpp
{


bool Status::ok () const
{
    return code == Code::Ok;
}

bool Status::error () const
{
    return std::size_t(code) >= std::size_t(Code::NotOk);
}

Status::operator bool() const
{
    return ok();
}

std::string Status::toString() const
{
    switch(code)
    {
        case Code::Ok:
            return "OK";
        default:
            return "";
    }
}

std::ostream& operator<< (std::ostream& out, const Status& status)
{
    return out << status.toString();
}

void Status::set (Status::Code newCode)
{
    code = newCode;
}

}