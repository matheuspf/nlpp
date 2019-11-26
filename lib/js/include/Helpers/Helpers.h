#pragma once

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <nlpp/helpers/helpers.hpp>


namespace nlpp_js
{

namespace impl
{
    template <class V = nlpp::Vec>
    void vec (emscripten::val a, V& v)
    {
        v.resize(a["length"].as<unsigned>());

        for(int i = 0; i < v.size(); ++i)
            v(i) = a[i].as<nlpp::impl::Scalar<V>>();
    }

    template <class V = nlpp::Vec>
    V vec (emscripten::val a)
    {
        V v;

        vec(a, v);

        return v;
    }


    template <class V>
    emscripten::val array (const V& v)
    {
        double* buffer = new double[v.size()];

        std::copy(v.data(), v.data() + v.size(), buffer);

        return emscripten::val(emscripten::typed_memory_view(v.size(), buffer));
    }

    // template <class V>
    // void array (const V& v, emscripten::val& a)
    // {
    //     double* buffer = new double[v.size()];

    //     std::copy(v.data(), v.data() + v.size(), buffer);

    //     a = emscripten::typed_memory_view((v.size(), buffer));
    // }

} // namespace impl

} // namespace nlpp_js