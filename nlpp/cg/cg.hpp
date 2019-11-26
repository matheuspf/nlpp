/** @file
 *  @brief Nonlinear Conjugate Gradient
 * 
 *  @details An implementation of the nonlinear CG algorithm, described in chapter 5 of NOCEDAL.
 * 
 * 	The nonlinear version of the CG algorithms does not make any assumption about the objective function 
 * 	(it only needs to be smooth). Also, it has a quite concise structure, only changing the way to calculate
 *  the scalar factor of the previous directions that should be added to the current gradient.
 * 
 *  The methods implemented are: Fletcher-Reeves (FR), Polak-Ribière and variants (PR, PR_abs, PR_Plus, FR_PR),
 * 	Hestenes-Stiefel (HS)
*/

#pragma once

#include "cg_dec.hpp"
#include "projections.hpp"

namespace nlpp
{

namespace impl
{

template <class Base_>
template <class Function, class V>
V CG<Base_>::optimize (Function f, V x)
{
    V fa, dir, fb(x.rows(), x.cols());

    impl::Scalar<V> fxOld, fx;

    std::tie(fxOld, fa) = f(x);

    dir = -fa;

    for(int iter = 0; iter < stop.maxIterations(); ++iter)
    {
        double alpha = lineSearch(f, x, dir);
    
        x = x + alpha * dir;

        fx = f(x, fb);
    
        if(stop(*this, x, fx, fb))
            break;
    
        if((fa.dot(fb) / fb.dot(fb)) >= v)
            dir = -fb;
    
        else
            dir = -fb + cg(fa, fb, dir) * dir;
    
        fxOld = fx;
    
        fa = fb;
    
        output(*this, x, fx, fb);
    }

    return x;
}

} // namespace impl

} // namespace nlpp