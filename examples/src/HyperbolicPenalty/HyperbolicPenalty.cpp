#include "HyperbolicPenalty/HyperbolicPenalty.h"

#include "CG/CG.h"

#include "TestFunctions/TreeBarTruss.h"


using namespace nlpp;



int main ()
{
    HyperbolicPenalty<> hp(10.0, 10.0);

    Vec x = hp(TreeBarTruss::func, TreeBarTruss::cons, Vec::Constant(2, 0.5));
    

    handy::print("x: ", x.transpose());
    handy::print("fx: ", TreeBarTruss::func(x));
    handy::print("cx: ", TreeBarTruss::cons(x).cwiseAbs().sum());


    return 0;
}