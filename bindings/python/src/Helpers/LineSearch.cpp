#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "LineSearch/LineSearch.h"
#include "LineSearch/StrongWolfe/StrongWolfe.h"
#include "LineSearch/Goldstein/Goldstein.h"


namespace py = pybind11;


class PyLineSearch : public nlpp::poly::LineSearch<> {
public:

    using Base = nlpp::poly::LineSearch<>;
    using Base::Base;


    Base::Float lineSearch (::nlpp::wrap::LineSearch<::nlpp::wrap::poly::FunctionGradient<Base::V>, Base::V> lsWrapper)
    {
        PYBIND11_OVERLOAD_PURE(
            Base::Float,
            Base,
            lineSearch,
            lsWrapper
        );
    }

    void initialize ()
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Base,
            initialize 
        );
    }

    Base* clone_impl () const
    {
        PYBIND11_OVERLOAD_PURE(
            Base*,
            Base,
            clone_impl 
        );
    }
};


void add_lineSearch (py::module& m)
{
    py::class_<nlpp::poly::LineSearch<>, PyLineSearch> lineSearchBase(m, "LineSearchBase");
        lineSearchBase
        .def(py::init<>())
        .def("lineSearch", &nlpp::poly::LineSearch<>::lineSearch);

    py::class_<nlpp::poly::LineSearch_<>>(m, "LineSearch")
        .def(py::init<>())
        .def(py::init([](nlpp::poly::LineSearch<>* ls){ return nlpp::poly::LineSearch_<>(ls->clone()); }))
        .def("lineSearch", static_cast<double (nlpp::poly::LineSearch_<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::LineSearch_<>::operator()));


    py::class_<nlpp::poly::StrongWolfe<>>(m, "StrongWolfe", lineSearchBase)
        .def(py::init<>())
        .def("lineSearch", static_cast<double (nlpp::poly::StrongWolfe<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::StrongWolfe<>::operator()));


    py::class_<nlpp::poly::Goldstein<>>(m, "Goldstein", lineSearchBase)
        .def(py::init<>())
        .def("lineSearch", static_cast<double (nlpp::poly::Goldstein<>::*)
                           (const std::function<double(const nlpp::Vec&)>&, const nlpp::Vec&, const nlpp::Vec&)>
                           (&nlpp::poly::Goldstein<>::operator()));


    py::implicitly_convertible<nlpp::poly::StrongWolfe<>, nlpp::poly::LineSearch_<>>();
    py::implicitly_convertible<nlpp::poly::Goldstein<>, nlpp::poly::LineSearch_<>>();
}
