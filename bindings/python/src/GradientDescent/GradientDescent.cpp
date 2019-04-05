#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "GradientDescent/GradientDescent.h"


namespace py = pybind11;


void add_gradientDescent(py::module& m)
{
    py::class_<nlpp::poly::GradientDescent<>>(m, "GradientDescent")
        .def(py::init<>())
        
        .def("optimize", static_cast<nlpp::Vec (nlpp::poly::GradientDescent<>::*)
                                     (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&)>
                         (&nlpp::poly::GradientDescent<>::operator()))

       .def("optimize", static_cast<nlpp::Vec (nlpp::poly::GradientDescent<>::*)
                                     (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const std::function<::nlpp::wrap::poly::FunctionGradient<>::GradType_1>&, const nlpp::Vec&)>
                         (&nlpp::poly::GradientDescent<>::operator()))
  
                         
        // .def("__call__", static_cast<nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                              (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&)>
        //                  (&nlpp::poly::GradientDescent<>::operator()), py::is_operator())
 

        .def_readwrite("stop", &nlpp::poly::GradientDescent<>::stop)
        .def_readwrite("output", &nlpp::poly::GradientDescent<>::output)
        .def_readwrite("lineSearch", &nlpp::poly::GradientDescent<>::lineSearch);
}
