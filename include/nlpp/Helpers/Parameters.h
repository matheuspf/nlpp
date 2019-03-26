#pragma once

#include "Helpers.h"


namespace nlpp
{

/// Parameters namespace
namespace params
{

template <class Stop_, class Output_>
struct Optimizer
{
    using Stop = Stop_;
    using Output = Output_;

    /** @name
     * @brief Some basic constructors
    */
    Optimizer (const Stop& stop = Stop{}, const Output& output = Output{}) : stop(stop), output(output)
    {
    }
    //@}


    void initialize () {}


    Stop stop;      ///< Stopping condition

    Output output;  ///< The output callback
};



/** @brief Base parameter class for gradient optimizers
 *  @details Define the basic variables used by any gradient based optimizer
*/
template <class LineSearch_, class Stop_, class Output_>
struct LineSearchOptimizer : public Optimizer<Stop_, Output_>
{
    using Base = Optimizer<Stop_, Output_>;

    using LineSearch = LineSearch_;
    using Stop = typename Base::Stop;
    using Output = typename Base::Output;


    /** @name
     * @brief Some basic constructors
    */
    LineSearchOptimizer (const LineSearch& lineSearch = LineSearch{}, const Stop& stop = Stop{}, const Output& output = Output{}) :
                         Base(stop, output), lineSearch(lineSearch)
    {
    }
    //@}


    LineSearch lineSearch;  ///< The line search method
};


} // namespace params

} // namespace nlpp