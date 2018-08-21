#pragma once

#include "Helpers.h"


namespace nlpp
{

/// Parameters namespace
namespace params
{

/** @brief Base parameter class for gradient optimizers
 * 
 *  @details Define the basic variables used by any gradient based optimizer
*/
template <class LineSearch_, class Stop_, class Output_>
struct GradientOptimizer
{
    using LineSearch = LineSearch_;
    using Stop = Stop_;
    using Output = Output_;


    /** @name
     * @brief Some basic constructors
    */
    GradientOptimizer(const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) :
                      lineSearch(lineSearch), stop(stop), output(output)
    {
    }
    //@}


    LineSearch lineSearch;  ///< The line search method

    Stop stop;      ///< Stopping condition

    Output output;  ///< The output callback
};


} // namespace params

} // namespace nlpp