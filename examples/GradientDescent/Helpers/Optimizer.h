#pragma once

#include "Helpers.h"


namespace js_nlp
{

template <class Impl>
struct Optimizer
{
    emscripten::val optimize (emscripten::val f, emscripten::val x)
    {
        res = static_cast<Impl&>(*this).operator()([&](const nlpp::Vec& x)
        {
            return f(emscripten::val(emscripten::typed_memory_view(x.size(), x.data()))).template as<double>();

        }, makeVec(x));

        return emscripten::val(emscripten::typed_memory_view(res.size(), res.data()));
    }

    nlpp::Vec res;
};



} // namespace js_nlp
