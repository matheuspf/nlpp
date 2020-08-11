#pragma once

#include "config.hpp"

#include <type_traits>
#include <memory>
#include <deque>

#include <eigen3/Eigen/Dense>

#include <handy/Handy.h>

#include <Spectra/SymEigsSolver.h>


#if NLPP_USE_NANOFLANN
#   include <nanoflann.hpp>
#endif
