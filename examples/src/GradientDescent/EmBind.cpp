#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "GradientDescent/GradientDescent.h"


using namespace emscripten;



struct JS_Out
{
    JS_Out (val f) : f(f) {}

    template <class Optimizer>
    void init (const Optimizer& optimizer, double fx)

    {
        f(fx);
    }

    template <class Optimizer>
    void operator () (const Optimizer& optimizer, double fx)
    {
        f(fx);     
    }

    template <class Optimizer>
    void finish (const Optimizer& optimizer, double fx)
    {
        f(fx);
    }


    val f;
};




struct GD : public nlpp::GradientDescent<nlpp::Goldstein, JS_Out>,
            public js_nlp::Optimizer<GD>
{
    using Base = nlpp::GradientDescent<nlpp::Goldstein, JS_Out>;

    GD(val js_out) : Base(nlpp::params::GradientDescent<nlpp::Goldstein, JS_Out>(nlpp::Goldstein{}, JS_Out(js_out)))
    {
    }

    using js_nlp::Optimizer<GD>::optimize;
};





EMSCRIPTEN_BINDINGS(Optimization) {
    emscripten::class_<GD>("GD")
        .constructor<val>()
        .function("optimize", static_cast<emscripten::val (GD::*)(emscripten::val, emscripten::val)>
                              (&GD::optimize));
}