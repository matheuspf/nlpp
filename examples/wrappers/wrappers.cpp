#include "utils/wrappers.hpp"
#include "utils/finiteDifference.hpp"


struct F1
{
    double function (const nlpp::Vec& x) const
    {
        return x.dot(x);
    }

    // double operator() (const nlpp::Vec& x) const
    // {
    //     return x.dot(x);
    // }
};

struct F2
{
    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x, nlpp::impl::Plain2D<V>& h) const
    {
    }

    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x, nlpp::impl::Plain2D<V>& h) const
    {
    }

    template <class V>
    auto hessian (const Eigen::MatrixBase<V>& x) const
    {
        return Eigen::MatrixXd(10, 10);
    }
    
    template <class V>
    auto operator() (const Eigen::MatrixBase<V>& x) const
    {
        return Eigen::MatrixXd(10, 10);
    }

};


int main ()
{
    auto func = nlpp::wrap::functionsBuilder<nlpp::wrap::Conditions::Function | nlpp::wrap::Conditions::Gradient | nlpp::wrap::Conditions::Hessian, nlpp::Vec>(F1{});
    // auto func = nlpp::wrap::functionsBuilder<nlpp::wrap::Conditions::Function | nlpp::wrap::Conditions::Gradient, nlpp::Vec>(F1{});

    using TFs = typename std::decay_t<decltype(func)>::TFs;

    using TF0 = std::tuple<std::tuple_element_t<0, TFs>>;
    using TF1 = std::tuple<std::tuple_element_t<1, TFs>>;
    using TF2 = std::tuple<std::tuple_element_t<2, TFs>>;
    
    // nlpp::impl::PrintType<TFs>{};

    constexpr auto id = nlpp::wrap::impl::OpId<nlpp::wrap::impl::IsGradient_1, nlpp::Vec, TFs>;
    constexpr auto id0 = nlpp::wrap::impl::OpId<nlpp::wrap::impl::IsGradient_1, nlpp::Vec, TF0>;
    constexpr auto id1 = nlpp::wrap::impl::OpId<nlpp::wrap::impl::IsGradient_1, nlpp::Vec, TF1>;
    constexpr auto id2 = nlpp::wrap::impl::OpId<nlpp::wrap::impl::IsGradient_1, nlpp::Vec, TF2>;

    // std::cout << id << "\t" << std::tuple_size<TFs>() <<  "\n";
    // std::cout << id0 << "\n";
    // std::cout << id1 << "\n";
    // std::cout << id2 << "\n";


    // std::cout << nlpp::wrap::impl::IsGradient_1<std::tuple_element_t<0, TFs>, nlpp::Vec>::value << "\n";
    // std::cout << nlpp::wrap::impl::IsGradient_1<std::tuple_element_t<1, TFs>, nlpp::Vec>::value << "\n";
    // std::cout << nlpp::wrap::impl::IsGradient_1<std::tuple_element_t<2, TFs>, nlpp::Vec>::value << "\n";


    // constexpr auto id3 = nlpp::wrap::impl::OpId<nlpp::wrap::impl::IsHessian_2, nlpp::Vec, TFs>;
    // constexpr auto id3 = nlpp::wrap::impl::HasOp<nlpp::wrap::impl::IsHessian_1, nlpp::Vec, TFs>;

    // std::cout << id3 << "\n";



    nlpp::Vec x0 = nlpp::Vec::Constant(2, 1.0);

    auto r1 = func.funcGrad(x0);
    std::cout << r1.first << "\t" << r1.second.transpose() << "\n\n";


    auto r2 = func.hessian(x0, -x0);
    std::cout << r2 << "\n\n";


    return 0;
}
