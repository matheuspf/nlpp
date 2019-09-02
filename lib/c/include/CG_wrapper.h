#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CG CG;

CG* new_cg();

void del_cg(CG*);

double opt_cg(CG*, double(*)(double*, int), double*, int);


#ifdef __cplusplus
}
#endif