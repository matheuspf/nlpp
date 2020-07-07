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

auto f5 = [](const auto& x)
{
    return std::make_pair(F1{}.ineqs(x), F2{}.eqs(x));
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
    auto c4 = constraints<Conditions::NLInequalities>(f4);
    auto c44 = constraints<Conditions::NLInequalities | Conditions::NLEqualities>(f4, f4);
    auto c5 = constraints<Conditions::NLInequalities | Conditions::NLEqualities>(f5);
    auto c6 = constraints<Conditions::Empty>();


    std::cout << c1.ineqs(x0).transpose() << "\n";
    std::cout << c2.eqs(x0).transpose() << "\n";
    std::cout << c12.ineqs(x0).transpose() << "\t" << c12.eqs(x0).transpose() << "\n";
    std::cout << c3.ineqs(x0).transpose() << "\n";
    std::cout << c33.ineqs(x0).transpose() << "\t" << c33.eqs(x0).transpose() << "\n";
    std::cout << c4.ineqs(x0).transpose() << "\n";
    std::cout << c44.ineqs(x0).transpose() << "\t" << c44.eqs(x0).transpose() << "\n";
    std::cout << std::get<0>(c44(x0)).transpose() << "\t" << std::get<1>(c44(x0)).transpose() << "\n";
    std::cout << c5.ineqs(x0).transpose() << "\t" << c5.ineqs(x0).transpose() << "\n";
    std::cout << std::get<0>(c5(x0)).transpose() << "\t" << std::get<1>(c5(x0)).transpose() << "\n";



    return 0;
}
