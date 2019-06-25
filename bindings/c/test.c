#include <stdio.h>
#include <stdlib.h>
#include "./include/CG_wrapper.h"


double func (double* x, int n)
{
    return (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5);
}

int main ()
{
    CG* cg = new_cg();

    double x[2] = {1.0, -1.0};

    double fx = opt_cg(cg, func, x, 2);

    printf("%.4f %.4f\n%.4f\n", x[0], x[1], fx);


    return 0;
}