#pragma once

#include "Helpers.h"


#define NLPP_JS_OPTIMIZER(Name) \
\
emscripten::class_<nlpp::poly::Name<>, emscripten::base<nlpp::poly::GradientOptimizer<>>>(#Name) \
    .constructor<>()        \
    .function("optimize", (emscripten::val (*)(nlpp::poly::GradientOptimizer<>&, emscripten::val, emscripten::val)) &nlpp_js::optimize<>)   \
    .function("optimize", (emscripten::val (*)(nlpp::poly::GradientOptimizer<>&, emscripten::val, emscripten::val, emscripten::val)) &nlpp_js::optimize<>);



namespace nlpp_js
{

template <class V = nlpp::Vec>
emscripten::val optimize (nlpp::poly::GradientOptimizer<V>& optimizer, emscripten::val js_function, emscripten::val js_x0)
{
    auto function = [&](const auto& x){ return js_function(impl::array(x)). template as<double>(); };

    auto x0 = impl::vec(js_x0);

    auto res = optimizer(function, x0);

    return impl::array(res);
}

template <class V = nlpp::Vec>
emscripten::val optimize (nlpp::poly::GradientOptimizer<V>& optimizer, emscripten::val js_function, emscripten::val js_gradient, emscripten::val js_x0)
{
    auto function = [&](const auto& x){ return js_function(impl::array(x)). template as<double>(); };

    auto gradient = [&](const nlpp::Vec& x) -> nlpp::Vec { return impl::vec(js_gradient(impl::array(x))); };

    auto x0 = impl::vec(js_x0);

    auto res = optimizer(function, gradient, x0);

    return impl::array(res);
}



template <class V = nlpp::Vec>
struct JS_Output : public nlpp::out::poly::GradientOptimizerBase<V>
{
    using Float = ::nlpp::impl::Scalar<V>;

    //JS_Output (emscripten::val init, emscripten::val opt, emscripten::val finish) {}
    JS_Output (emscripten::val f) : f(f) {}


    virtual void init (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        //Impl::init(optimizer, x, fx, gx);
    }

    virtual void operator() (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        f(impl::array(x), emscripten::val(fx), impl::array(gx));
    }

    virtual void finish (const nlpp::params::poly::LineSearchOptimizer_& optimizer, const Eigen::Ref<const V>& x, Float fx, const Eigen::Ref<const V>& gx)
    {
        //Impl::finish(optimizer, x, fx, gx);
    }

    virtual JS_Output* clone_impl () const { return new JS_Output(*this); }

    emscripten::val f;
};


template <class V = nlpp::Vec>
void getOutput (nlpp::poly::GradientOptimizer<V>& optimizer, emscripten::val js_function)
{
    optimizer.output = new JS_Output<V>(js_function);
}

} // namespace nlpp_js