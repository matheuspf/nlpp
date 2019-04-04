#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Wrappers.h"


namespace py = pybind11;

void add_wrappers (py::module& m)
{
    py::class_<::nlpp::wrap::poly::Function<>>(m, "Function")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&>());
        
    py::class_<::nlpp::wrap::poly::FunctionGradient<>>(m, "FunctionGradient")
        .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&>());
}


// PYBIND11_MODULE(nlpy, m)
// {
    // py::class_<::nlpp::wrap::poly::Function<>>(m, "Function")
    //     .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&>());
        
    // py::class_<::nlpp::wrap::poly::FunctionGradient<>>(m, "FunctionGradient")
    //     .def(py::init<const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncGradType_1>&>());
// }