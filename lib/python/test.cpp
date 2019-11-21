#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "lib/cpp/include/cg/cg.hpp"


namespace py = pybind11;


template <class V = ::nlpp::Vec>
class PyGradientOptimizer : public nlpp::poly::LineSearchOptimizer<V> {
public:
    using nlpp::poly::LineSearchOptimizer<>::LineSearchOptimizer;

    V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
    {
        PYBIND11_OVERLOAD_PURE(V, nlpp::poly::LineSearchOptimizer<V>, optimize, f, x);
    }

    nlpp::poly::LineSearchOptimizer<V>* clone_impl () const
    {
        PYBIND11_OVERLOAD_PURE(nlpp::poly::LineSearchOptimizer<V>*, nlpp::poly::LineSearchOptimizer<V>, clone_impl,);
    }
};


PYBIND11_MODULE(nlpy, m)
{
    py::class_<nlpp::poly::LineSearchOptimizer<>, PyGradientOptimizer<>> gradientOptimizer(m, "GradientOptimizer");

    gradientOptimizer
        .def(py::init<>())
        .def("optimize", (nlpp::Vec (nlpp::poly::LineSearchOptimizer<>::*)(nlpp::wrap::poly::FunctionGradient<>, nlpp::Vec))&nlpp::poly::LineSearchOptimizer<>::optimize)
        .def_readwrite("stop", &nlpp::poly::LineSearchOptimizer<>::stop)
        .def_readwrite("output", &nlpp::poly::LineSearchOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::LineSearchOptimizer<>::lineSearch);

    py::class_<nlpp_p::CG<>>(m, "CG")
        .def(py::init<>())
        
        .def("optimize", (nlpp::Vec (nlpp_p::CG<>::*)(nlpp::wrap::poly::FunctionGradient<>, nlpp::Vec))&nlpp_p::CG<>::optimize)

        // .def("__call__", (nlpp::Vec (nlpp_p::CG<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&))
        //                  &nlpp_p::CG<>::operator(), py::is_operator())

        // .def("__call__", (nlpp::Vec (nlpp_p::CG<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const std::function<::nlpp::wrap::poly::FunctionGradient<>::GradType_1>&, const nlpp::Vec&))
        //                  &nlpp_p::CG<>::operator(), py::is_operator())


        .def("__call__", (nlpp::Vec (nlpp_p::CG<>::*)
                         (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&, const nlpp::Vec&))
                         &nlpp_p::CG<>::operator(), py::is_operator())
        

        .def_readwrite("stop", &nlpp_p::CG<>::stop)
        .def_readwrite("output", &nlpp::poly::LineSearchOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::LineSearchOptimizer<>::lineSearch);


    py::class_<::nlpp::wrap::poly::Function<>>(m, "Function")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&>());
        
    py::class_<::nlpp::wrap::poly::FunctionGradient<>>(m, "FunctionGradient")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&>());

    //py::implicitly_convertible<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, ::nlpp::wrap::poly::FunctionGradient<>>();

}