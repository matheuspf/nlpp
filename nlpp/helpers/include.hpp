#pragma once

#include "config.hpp"

#include <cstdlib>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <deque>

#ifndef _MSC_VER
#   include <cxxabi.h>
#endif

#include <Eigen/Dense>

#include <handy/Handy.h>

#include <Spectra/SymEigsSolver.h>


#if NLPP_USE_NANOFLANN
#   include <nanoflann.hpp>
#endif
