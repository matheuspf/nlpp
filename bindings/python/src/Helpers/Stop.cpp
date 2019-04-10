#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Stop.h"


namespace py = pybind11;


void add_lineSearch(py::module& m)
{
    py::class_<nlpp::stop::poly::GradientOptimizer_<>>(m, "Stop").
        def(py::init<>()).
        def(py::init<nlpp::stop::poly::GradientOptimizer<true>>()).
        def(py::init<nlpp::stop::poly::GradientOptimizer<false>>()).
        def(py::init<nlpp::stop::poly::GradientNorm<>>()).
        def("stop", &nlpp::stop::poly::GradientOptimizer_<>::operator()).
        def("set", static_cast<nlpp::stop::poly::GradientOptimizer_<>& (nlpp::stop::poly::GradientOptimizer_<>::*)()>(&nlpp::stop::poly::GradientOptimizer_<>::set)).
        def("set", static_cast<void (nlpp::stop::poly::GradientOptimizer_<>::*)(int)>(&nlpp::stop::poly::GradientOptimizer_<>::set)).
        def("set", static_cast<void (nlpp::stop::poly::GradientOptimizer_<>::*)(std::string)>(&nlpp::stop::poly::GradientOptimizer_<>::set));


    py::implicitly_convertible<nlpp::poly::StrongWolfe<>, nlpp::poly::LineSearch_<>>();
    py::implicitly_convertible<nlpp::poly::Goldstein<>, nlpp::poly::LineSearch_<>>();
}
