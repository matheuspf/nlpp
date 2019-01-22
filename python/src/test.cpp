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
        .def("optimize", &nlpp::poly::GradientOptimizer<>::optimize)
        .def_readwrite("stop", &nlpp::poly::GradientOptimizer<>::stop)
        .def_readwrite("output", &nlpp::poly::GradientOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::GradientOptimizer<>::lineSearch);

    py::class_<nlpp::poly::GradientDescent<>>(m, "GradientDescent")
        .def(py::init<>())
        .def("optimize", &nlpp::poly::GradientOptimizer<>::optimize)
        //.def("__call__", [](const std::function<double(const Eigen::Ref<const nlpp::Vec>&)> function, const Eigen::Ref<const nlpp::Vec>& x){  })
        .def_readwrite("stop", &nlpp::poly::GradientDescent<>::stop)
        .def_readwrite("output", &nlpp::poly::GradientOptimizer<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::GradientOptimizer<>::lineSearch);


    py::class_<::nlpp::wrap::poly::FunctionGradient<>>(m, "Function")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&>());

    py::implicitly_convertible<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, ::nlpp::wrap::poly::FunctionGradient<>>();


    // py::class_<nlpp::out::poly::GradientOptimizer_<>>(m, "Output")




    // py::class_<Test>(m, "Test")
    //     .def(py::init<const std::string&>(), py::arg("x") = "abc")
    //     .def_readwrite("x", &Test::x);

    // // py::class_<nlpp::StrongWolfe<double>>(m, "StrongWolfe")
    // //     .def(py::init<double, double, double>(), py::arg("a0") = 1.0, py::arg("c1") = 1e-4, py::arg("c2") = 0.9)
    // //     .def("lineSearch", &nlpp::StrongWolfe<double>::lineSearch<std::function<double(const nlpp::Vec)>>);

    // py::class_<Func>(m, "Func")
    //     .def(py::init<>())
    //     .def("func", &Func::operator()); 

    // py::class_<nlpp::poly::StrongWolfe<Func, double>>(m, "StrongWolfe")
    //     .def(py::init<double, double, double>(), py::arg("a0") = 1.0, py::arg("c1") = 1e-4, py::arg("c2") = 0.9)
    //     .def("lineSearch", &nlpp::poly::StrongWolfe<Func, double>::lineSearch);

}