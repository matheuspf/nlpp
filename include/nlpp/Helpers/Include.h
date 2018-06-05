#pragma once


#ifndef EIGEN_INCLUDE_LOCAL
    #define EIGEN_INCLUDE_LOCAL 0
#endif

#define EIGEN_INCLUDE_LOCAL_RELEASE 0
#define EIGEN_INCLUDE_GLOBAL 1


#if EIGEN_INCLUDE_LOCAL

    #include "Eigen/Eigen/Dense"

#elif EIGEN_INCLUDE_LOCAL_RELEASE

    #include "Eigen/Dense"

#elif EIGEN_INCLUDE_GLOBAL

    //#include <Eigen/Dense>

#endif



#define HANDY_INCLUDE_LOCAL 1
#define HANDY_INCLUDE_LOCAL_RELEASE 0
#define HANDY_INCLUDE_GLOBAL 0


#if HANDY_INCLUDE_GLOBAL

    #include <Handy/Handy.h>

#elif HANDY_INCLUDE_LOCAL_RELEASE

    #include "Handy/Handy.h"

#elif HANDY_INCLUDE_LOCAL

    #include "handy/include/Handy.h"

#endif
