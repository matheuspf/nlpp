#include "CG/CG.h"
#include "CG_wrapper.h"


CG* new_cg ()
{
    return reinterpret_cast<CG*>(new nlpp::CG<>());
}

void del_cg (CG* cg)
{
    delete reinterpret_cast<nlpp::CG<>*>(cg);
}

double opt_cg (CG* _cg, double (*_func)(double*, int), double* _x, int _n)
{
    nlpp::CG<>& cg = *reinterpret_cast<nlpp::CG<>*>(_cg);

    auto func = [_func, _n](const nlpp::Vec& x){
        return _func((double*)&x(0), _n);
        
    };

    nlpp::Vec x(_n);

    for(int i = 0; i < _n; ++i)
        x(i) = _x[i];

    x = cg(func, x);

    for(int i = 0; i < _n; ++i)
        _x[i] = x(i);

    return _func(_x, _n);
}