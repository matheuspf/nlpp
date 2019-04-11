#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Helpers.h"


namespace py = pybind11;


void add_lineSearch(py::module&);
void add_stop(py::module&);
void add_output(py::module&);
void add_gradientDescent(py::module&);
void add_CG(py::module&);



PYBIND11_MODULE(nlpy, m)
{
    add_lineSearch(m);
    add_stop(m);
    add_output(m);
    add_gradientDescent(m);
    add_CG(m);
}