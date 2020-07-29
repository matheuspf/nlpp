#pragma once

#include "config.hpp"

#include <type_traits>
#include <memory>
#include <deque>


#if EIGEN_INCLUDE_LOCAL

    #include "../external/Eigen/Eigen/Dense"

#elif EIGEN_INCLUDE_LOCAL_RELEASE

    #include "Eigen/Dense"

#elif EIGEN_INCLUDE_GLOBAL

    #include <eigen3/Eigen/Dense>

#endif


#if HANDY_INCLUDE_GLOBAL

    #include <handy/Handy.h>

#elif HANDY_INCLUDE_LOCAL_RELEASE

    #include "handy/Handy.h"

#elif HANDY_INCLUDE_LOCAL

    #include "../external/handy/include/handy/Handy.h"

#endif


#if SPECTRA_INCLUDE_GLOBAL

    #include <Spectra/SymEigsSolver.h>

#elif SPECTRA_INCLUDE_LOCAL_RELEASE

    #include "Spectra/SymEigsSolver.h"

#elif SPECTRA_INCLUDE_LOCAL

    #include "../external/spectra/include/Spectra/SymEigsSolver.h"

#endif


#if USE_NANOFLANN
    #if NANOFLANN_INCLUDE_GLOBAL

        #include <nanoflann/nanoflann.hpp>

    #elif NANOFLANN_INCLUDE_LOCAL_RELEASE

        #include "nanoflann/nanoflann.hpp"

    #elif NANOFLANN_INCLUDE_LOCAL

        #include "../external/nanoflann/include/nanoflann.hpp"

    #endif
#endif