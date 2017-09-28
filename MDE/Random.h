/** \file Random.h
  *
  * Simple helper classes to generate random uniform integer and 
  * real numbers. Example:
  * 
  * RandDouble randDouble;
  * randDouble(0.0, 1.0);    // random number between 0.0 and 1.0   
  *
  * Notice that the RandInt is open in the right, so 'max' must be 
  * greater than 'min'.
  *
*/

#ifndef RANDOM_HELPER_H
#define RANDOM_HELPER_H

#include <random>

namespace help
{
    struct Rand
    {
        Rand () : generator(std::random_device{}()) {}

        std::mt19937 generator;
    };


    struct RandInt : Rand
    {
        inline int operator() (int min, int max)
        {
            return std::uniform_int_distribution<>(min, max - 1)(generator);
        }
    };


    struct RandDouble : Rand
    {
        inline double operator() (double min, double max)
        {
            return std::uniform_real_distribution<>(min, max)(generator);
        }
    };
}

#endif //RANDOM_HELPER_H
