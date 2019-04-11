#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "CG/CG.h"


namespace py = pybind11;


void add_CG(py::module& m)
{
    py::class_<nlpp::poly::CG<>>(m, "CG")
        .def(py::init<>())
        .def(py::init<const nlpp::poly::LineSearch_<>&>())
        .def(py::init<const nlpp::poly::LineSearch_<>&, const nlpp::stop::poly::GradientOptimizer_<>&>())
        .def(py::init<const nlpp::poly::LineSearch_<>&, const nlpp::stop::poly::GradientOptimizer_<>&, const nlpp::out::poly::GradientOptimizer_<>&>())
        
        .def("optimize", static_cast<nlpp::Vec (nlpp::poly::CG<>::*)
                                     (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&)>
                         (&nlpp::poly::CG<>::operator()))

        .def("optimize", static_cast<nlpp::Vec (nlpp::poly::CG<>::*)
                                     (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const std::function<::nlpp::wrap::poly::FunctionGradient<>::GradType_1>&, const nlpp::Vec&)>
                         (&nlpp::poly::CG<>::operator()))

        .def_readwrite("stop", &nlpp::poly::CG<>::stop)
        .def_readwrite("output", &nlpp::poly::CG<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::CG<>::lineSearch);
}
