#include "utils/wrappers/functions.hpp"
#include "utils/finite_difference_dec.hpp"


struct F1
{
    template <class V>
    nlpp::impl::Scalar<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return x[0] * x[0];
    }
};



int main ()
{
    using V = Eigen::Matrix<float, 2, 1>;

    auto func = nlpp::wrap::fd::functions<nlpp::wrap::Conditions::Function | nlpp::wrap::Conditions::Gradient | nlpp::wrap::Conditions::Hessian, V>(F1{});


    V x0 = V::Constant(2, 1.0);

    auto r1 = func.funcGrad(x0);
    std::cout << r1.first << "\t" << r1.second.transpose() << "\n\n";

    auto r2 = func.hessian(x0);
    std::cout << r2 << "\n\n";

    auto r3 = func.gradientDir(x0, -x0);
    std::cout << r3 << "\n\n";


    return 0;
}
