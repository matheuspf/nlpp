#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "Helpers/Helpers.h"


namespace py = pybind11;

// void add_wrappers(py::module&);
void add_gradientDescent(py::module&);
void add_CG(py::module&);
void add_lineSearch(py::module&);



PYBIND11_MODULE(nlpy, m)
{
    // add_wrappers(m);
    add_gradientDescent(m);
    add_CG(m);
    add_lineSearch(m);
}