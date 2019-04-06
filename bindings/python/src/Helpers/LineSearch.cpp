#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "LineSearch/LineSearch.h"
#include "LineSearch/StrongWolfe/StrongWolfe.h"
#include "LineSearch/Goldstein/Goldstein.h"


namespace py = pybind11;


void add_lineSearch(py::module& m)
{
    py::class_<nlpp::poly::LineSearch_<>>(m, "LineSearch")
        .def(py::init<>())
        .def(py::init<nlpp::poly::StrongWolfe<>>())
        .def(py::init<nlpp::poly::Goldstein<>>())
        .def("lineSearch", static_cast<double (nlpp::poly::LineSearch_<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::LineSearch_<>::operator()));


    py::class_<nlpp::poly::StrongWolfe<>>(m, "StrongWolfe")
        .def(py::init<>())
        .def("lineSearch", static_cast<double (nlpp::poly::StrongWolfe<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::StrongWolfe<>::operator()));


    py::class_<nlpp::poly::Goldstein<>>(m, "Goldstein")
        .def(py::init<>())
        .def("lineSearch", static_cast<double (nlpp::poly::Goldstein<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::Goldstein<>::operator()));


    py::implicitly_convertible<nlpp::poly::StrongWolfe<>, nlpp::poly::LineSearch_<>>();
    py::implicitly_convertible<nlpp::poly::Goldstein<>, nlpp::poly::LineSearch_<>>();
}
