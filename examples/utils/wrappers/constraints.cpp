#include "utils/wrappers/constraints.hpp"

using namespace nlpp;


struct F1
{
    template <class V>
    impl::Plain<V> ineqs (const Eigen::MatrixBase<V>& x) const
    {
        return x - impl::Plain<V>::Constant(x.rows(), 1.0);
    }
};

struct F2
{
    template <class V>
    impl::Plain<V> eqs (const Eigen::MatrixBase<V>& x) const
    {
        return x - impl::Plain<V>::Constant(x.rows(), 2.0);
    }
};

struct F3
{
    template <class V>
    impl::Plain<V> operator() (const Eigen::MatrixBase<V>& x) const
    {
        return x - impl::Plain<V>::Constant(x.rows(), 3.0);
    }
};

auto f4 = [](const auto& x)
{
    return x - impl::Plain<std::decay_t<decltype(x)>>::Constant(x.rows(), 4.0);
};


int main ()
{
    using V = Vec;
    using wrap::constraints, wrap::Constraints, wrap::Conditions;

    V x0 = V::Constant(2, 5.0);

    auto c1 = constraints<Conditions::NLInequalities>(F1{});
    auto c2 = constraints<Conditions::NLEqualities>(F2{});
    auto c12 = constraints<Conditions::NLInequalities | Conditions::NLEqualities>(F1{}, F2{});
    auto c3 = constraints<Conditions::NLInequalities>(F3{});
    auto c33 = constraints<Conditions::NLInequalities | Conditions::NLEqualities>(F3{}, F3{});
    auto c4_1 = constraints<Conditions::NLInequalities>(f4);
    auto c4_2 = constraints<Conditions::NLEqualities>(f4);


    std::cout << c1.ineqs(x0).transpose() << "\n";
    std::cout << c2.eqs(x0).transpose() << "\n";
    std::cout << c12.ineqs(x0).transpose() << "\t" << c12.eqs(x0).transpose() << "\n";
    std::cout << c3.ineqs(x0).transpose() << "\n";
    std::cout << c33.ineqs(x0).transpose() << "\t" << c33.eqs(x0).transpose() << "\n";
    std::cout << c4_1.ineqs(x0).transpose() << "\n";
    std::cout << c4_2.eqs(x0).transpose() << "\n";




    return 0;
}
