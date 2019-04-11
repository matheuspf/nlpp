#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Parameters.h"
#include "Helpers/Stop.h"
#include "Helpers/Output.h"


namespace py = pybind11;


class PyStopBase : public nlpp::stop::poly::GradientOptimizerBase<> {
public:

    using Base = nlpp::stop::poly::GradientOptimizerBase<>;
    using Base::Base;
    using V = Base::V;
    using Float = Base::Float;


    bool operator() (const nlpp::params::poly::Optimizer_& optimizer, const V& x, Float fx, const V& gx)
    {
        PYBIND11_OVERLOAD_PURE(
            bool,
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

    int maxIterations ()
    {
        PYBIND11_OVERLOAD_PURE(
            int,
            Base,
            maxIterations
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


void add_stop (py::module& m)
{
    py::class_<nlpp::stop::poly::GradientOptimizerBase<>, PyStopBase> stopBase(m, "StopBase");
        stopBase
        .def(py::init<>())
        .def("stop", &nlpp::stop::poly::GradientOptimizerBase<>::operator());


    py::class_<nlpp::stop::poly::GradientOptimizer_<>>(m, "Stop")
        .def(py::init<>())
        .def(py::init([](nlpp::stop::poly::GradientOptimizerBase<>* ls){ return nlpp::stop::poly::GradientOptimizer_<>(ls->clone()); }))
        .def("stop", &nlpp::stop::poly::GradientOptimizer_<>::operator());


    py::class_<nlpp::stop::poly::GradientOptimizer<true>>(m, "StopExclusive", stopBase)
        .def(py::init<>())
        .def("stop", &nlpp::stop::poly::GradientOptimizer<true>::operator());

    py::class_<nlpp::stop::poly::GradientOptimizer<false>>(m, "StopInclusive", stopBase)
        .def(py::init<>())
        .def("stop", &nlpp::stop::poly::GradientOptimizer<false>::operator());

    py::class_<nlpp::stop::poly::GradientNorm<>>(m, "StopGradNorm", stopBase)
        .def(py::init<>())
        .def("stop", &nlpp::stop::poly::GradientNorm<>::operator());


    py::implicitly_convertible<nlpp::stop::poly::GradientOptimizer<true>, nlpp::stop::poly::GradientOptimizer_<>>();
    py::implicitly_convertible<nlpp::stop::poly::GradientOptimizer<false>, nlpp::stop::poly::GradientOptimizer_<>>();
    py::implicitly_convertible<nlpp::stop::poly::GradientNorm<>, nlpp::stop::poly::GradientOptimizer_<>>();
}
