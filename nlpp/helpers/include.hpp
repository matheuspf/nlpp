#pragma once

#include "config.hpp"

#include <type_traits>
#include <memory>
#include <deque>

#include <Eigen/Dense>

#include <Handy.h>

#include <Spectra/SymEigsSolver.h>


#if NLPP_USE_NANOFLANN
#   include <nanoflann.hpp>
#endif
