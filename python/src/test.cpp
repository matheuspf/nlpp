#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "GradientDescent/GradientDescent.h"


namespace py = pybind11;



template <class V = ::nlpp::Vec>
class PyGradientOptimizer : public nlpp::poly::GradientOptimizer<V> {
public:
    using nlpp::poly::GradientOptimizer<>::GradientOptimizer;

    V optimize (::nlpp::wrap::poly::FunctionGradient<V> f, V x)
    {
        PYBIND11_OVERLOAD_PURE(V, nlpp::poly::GradientOptimizer<V>, optimize, f, x);
    }

    nlpp::poly::GradientOptimizer<V>* clone_impl () const
    {
        PYBIND11_OVERLOAD_PURE(nlpp::poly::GradientOptimizer<V>*, nlpp::poly::GradientOptimizer<V>, clone_impl,);
    }
};


PYBIND11_MODULE(nlpy, m)
{
    py::class_<nlpp::poly::GradientOptimizer<>, PyGradientOptimizer<>> gradientOptimizer(m, "GradientOptimizer");

    gradientOptimizer
        .def(py::init<>())
        .def("optimize", (nlpp::Vec (nlpp::poly::GradientOptimizer<>::*)(nlpp::wrap::poly::FunctionGradient<>, nlpp::Vec))&nlpp::poly::GradientOptimizer<>::optimize)
        .def_readwrite("stop", &nlpp::poly::GradientOptimizer<>::stop)
        .def_readwrite("output", &nlpp::poly::GradientOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::GradientOptimizer<>::lineSearch);

    py::class_<nlpp::poly::GradientDescent<>>(m, "GradientDescent")
        .def(py::init<>())
        
        .def("optimize", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)(nlpp::wrap::poly::FunctionGradient<>, nlpp::Vec))&nlpp::poly::GradientDescent<>::optimize)

        // .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&))
        //                  &nlpp::poly::GradientDescent<>::operator(), py::is_operator())

        // .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const std::function<::nlpp::wrap::poly::FunctionGradient<>::GradType_1>&, const nlpp::Vec&))
        //                  &nlpp::poly::GradientDescent<>::operator(), py::is_operator())


        .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
                         (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&, const nlpp::Vec&))
                         &nlpp::poly::GradientDescent<>::operator(), py::is_operator())
        

        .def_readwrite("stop", &nlpp::poly::GradientDescent<>::stop)
        .def_readwrite("output", &nlpp::poly::GradientOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::GradientOptimizer<>::lineSearch);


    py::class_<::nlpp::wrap::poly::Function<>>(m, "Function")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&>());
        
    py::class_<::nlpp::wrap::poly::FunctionGradient<>>(m, "FunctionGradient")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&>());

    //py::implicitly_convertible<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, ::nlpp::wrap::poly::FunctionGradient<>>();

}