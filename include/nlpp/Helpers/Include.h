#pragma once

#include "Config.h"


#if EIGEN_INCLUDE_LOCAL

    #include "../external/Eigen/Eigen/Dense"

#elif EIGEN_INCLUDE_LOCAL_RELEASE

    #include "Eigen/Dense"

#elif EIGEN_INCLUDE_GLOBAL

    #include <eigen3/Eigen/Dense>

#endif


#if HANDY_INCLUDE_GLOBAL

    #include <Handy/Handy.h>

#elif HANDY_INCLUDE_LOCAL_RELEASE

    #include "Handy/Handy.h"

#elif HANDY_INCLUDE_LOCAL

    #include "../external/handy/include/Handy.h"

#endif
