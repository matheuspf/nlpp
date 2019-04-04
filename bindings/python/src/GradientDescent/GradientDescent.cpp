#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "GradientDescent/GradientDescent.h"


namespace py = pybind11;


struct Test
{
    Eigen::VectorXd optimize (std::function<double(const Eigen::VectorXd&)> func, Eigen::VectorXd x)
    {
        //return x;
        return opt(func, x);
    }

    nlpp::GradientDescent<> opt;
};



void add_gradientDescent(py::module& m)
{
    py::class_<Test>(m, "Test")
        .def(py::init<>())
        .def("optimize", &Test::optimize);

    py::class_<nlpp::GradientDescent<>>(m, "GradientDescent")
        .def(py::init<>())
        
        // .def("optimize", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)(nlpp::wrap::poly::FunctionGradient<>, nlpp::Vec))&nlpp::poly::GradientDescent<>::optimize)

        // .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&))
        //                  &nlpp::poly::GradientDescent<>::operator(), py::is_operator())

        // .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const std::function<::nlpp::wrap::poly::FunctionGradient<>::GradType_1>&, const nlpp::Vec&))
        //                  &nlpp::poly::GradientDescent<>::operator(), py::is_operator())


        // .def("__call__", (nlpp::Vec (nlpp::poly::GradientDescent<>::*)
        //                  (const std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>&, const nlpp::Vec&))
        //                  &nlpp::poly::GradientDescent<>::operator(), py::is_operator())
        
        .def("optimize", (nlpp::Vec (nlpp::GradientDescent<>::*)
                         (std::function<::nlpp::wrap::poly::FunctionGradient<>::FuncType>, nlpp::Vec))
                         &nlpp::GradientDescent<>::operator());
 

        // .def_readwrite("stop", &nlpp::poly::GradientDescent<>::stop)
        // .def_readwrite("output", &nlpp::poly::GradientDescent<>::output)
        // .def_readwrite("lineSearch", &nlpp::poly::GradientDescent<>::lineSearch);
}