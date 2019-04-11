#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Parameters.h"
#include "Helpers/Stop.h"
#include "Helpers/Output.h"


namespace py = pybind11;


class PyOutputBase : public nlpp::out::poly::GradientOptimizerBase<> {
public:

    using Base = nlpp::out::poly::GradientOptimizerBase<>;
    using Base::Base;
    using V = Base::V;
    using Float = Base::Float;


    void operator() (const nlpp::params::poly::Optimizer_& optimizer, const V& x, Float fx, const V& gx)
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            Base,
            operator(),
            optimizer, x, fx, gx
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


void add_output (py::module& m)
{
    py::class_<nlpp::out::poly::GradientOptimizerBase<>, PyOutputBase> outputBase(m, "OutputBase");
        outputBase
        .def(py::init<>())
        .def("output", &nlpp::out::poly::GradientOptimizerBase<>::operator());


    py::class_<nlpp::out::poly::GradientOptimizer_<>>(m, "Output")
        .def(py::init<>())
        .def(py::init([](nlpp::out::poly::GradientOptimizerBase<>* ls){ return nlpp::out::poly::GradientOptimizer_<>(ls->clone()); }))
        .def("output", &nlpp::out::poly::GradientOptimizer_<>::operator());


    py::class_<nlpp::out::poly::GradientOptimizer<0>>(m, "OutputQuiet", outputBase)
        .def(py::init<>())
        .def("output", &nlpp::out::poly::GradientOptimizer<0>::operator());

    py::class_<nlpp::out::poly::GradientOptimizer<1>>(m, "OutputComplete", outputBase)
        .def(py::init<>())
        .def("output", &nlpp::out::poly::GradientOptimizer<1>::operator());

    py::class_<nlpp::out::poly::GradientOptimizer<2>>(m, "OutputStore", outputBase)
        .def(py::init<>())
        .def("output", &nlpp::out::poly::GradientOptimizer<2>::operator());


    py::implicitly_convertible<nlpp::out::poly::GradientOptimizer<0>, nlpp::out::poly::GradientOptimizer_<>>();
    py::implicitly_convertible<nlpp::out::poly::GradientOptimizer<1>, nlpp::out::poly::GradientOptimizer_<>>();
    py::implicitly_convertible<nlpp::out::poly::GradientOptimizer<2>, nlpp::out::poly::GradientOptimizer_<>>();
}
