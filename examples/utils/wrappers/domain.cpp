#include "utils/wrappers/domain.hpp"

using namespace nlpp;



int main ()
{
    using V = Vec;
    using V2 = Eigen::Vector2f;

    V x0 = V::Constant(2, 1.0);
    V lb = V::Constant(2, 0.0);
    V ub = V::Constant(2, 10.0);
    impl::Plain2D<V> A = impl::Plain2D<V>::Constant(2, 2, 1.0);
    V b = V::Constant(2, 10.0);
    impl::Plain2D<V> Aeq = impl::Plain2D<V>::Constant(2, 2, 1.0);
    V beq = V::Constant(2, 10.0);


    wrap::Domain<V, wrap::Conditions::Start> d1(x0);
    wrap::Domain<V, wrap::Conditions::Start | wrap::Conditions::Bounds> d2(x0, lb, ub);
    wrap::Domain<V, wrap::Conditions::LinearInequalities> d3(Eigen::MatrixXd(2, 2), b);

    std::cout << d1.x0.transpose() << "\n";
    std::cout << d2.lb.transpose() << "\n";
    std::cout << d3.b.transpose() << "\n";


    return 0;
}
