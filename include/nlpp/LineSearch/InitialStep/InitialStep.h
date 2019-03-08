#pragma once

#include "Helpers/Helpers.h"


namespace nlpp
{

struct InitialStepBase
{
    template <class LineSearch, class Stop, class Output, class V>
    void initialize (const params::GradientOptimizer<LineSearch, Stop, Output>& optimizer,
               	     const Eigen::MatrixBase<V>& x, typename V::Scalar fx, const Eigen::MatrixBase<V>& gx) 
	{
	}
};


} // namespace nlpp