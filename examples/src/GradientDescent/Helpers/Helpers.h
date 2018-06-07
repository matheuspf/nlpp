#pragma once

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "../../../../include/nlpp/Helpers/Helpers.h"

namespace js_nlp
{

nlpp::Vec makeVec (emscripten::val a)
{
    nlpp::Vec vec(a["length"].as<unsigned>());

    for(int i = 0; i < vec.size(); ++i)
        vec(i) = a[i].as<double>();

    return vec;
}


} // namespace js_nlp