#include "HyperbolicPenalty.h"
#include "../CG/CG.h"
#include "../TestFunctions/TreeBarTruss.h"




struct TestProblem
{
    static double func (const Vec& x)
    {
        return 4*pow(x(0) - 4, 2) + 25*pow(x(1) + 5, 2);
    }

    static Vec cons (const Vec& x)
    {
        Vec r(3);

        r(0) = -x(0);
        r(1) = -x(1);
        r(2) = (pow(x[0], 2) + pow(x[1], 2)) - 1;

        return r;
    }
};


int main ()
{
    HyperbolicPenalty<BFGS<StrongWolfe, BFGS_Constant>> hp(10.0, 10.0);

    Vec x = hp(TreeBarTruss::func, TreeBarTruss::cons, Vec::Constant(2, 0.5));
    
    //Vec x = hp(TestProblem::func, TestProblem::cons, Vec::Constant(2, 0.0));

    db(x.transpose());
    //db(TreeBarTruss::func(x), "       ", TreeBarTruss::cons(x).transpose());

    return 0;
}