#include "HyperbolicPenalty/HyperbolicPenalty.h"

#include "CG/CG.h"

#include "TestFunctions/TreeBarTruss.h"


using namespace nlpp;



int main ()
{
    HyperbolicPenalty<BFGS<StrongWolfe, BFGS_Constant>> hp(10.0, 10.0);

    Vec x = hp(TreeBarTruss::func, TreeBarTruss::cons, Vec::Constant(2, 0.5));
    

    handy::print(x.transpose());


    return 0;
}