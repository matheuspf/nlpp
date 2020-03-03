#include "cg/cg.hpp"
#include "TestFunctions/Rosenbrock.h"


// template <class Impl, class Func, class V>
// void exec (nlpp::LineSearchOptimizer<Impl>& opt, const Func& func, const Eigen::MatrixBase<V>& x0)
// {
//     nlpp::Vec res;
    
//     handy::print(handy::benchmark([&]{
//         res = opt(func, x0);
//     }));

//     handy::print(res.transpose());

// }


template <class T, class... Args> \
using TestInvoke = decltype(std::declval<T>().funcGrad(std::declval<Args>()...));


int main ()
{
    // nlpp::CG<> opt;
    // nlpp::Rosenbrock func;
    // nlpp::Vec x = nlpp::Vec::Constant(10, 2.0);
    // // Eigen::Vector4d x; x[0] = x[1] = x[2] = x[3] = 2.0;
    // // nlpp::Vec x(4); x[0] = x[1] = x[2] = x[3] = 2.0;

    // handy::print(handy::benchmark([&]{
    //     x = opt(func, x);
    // }));

    // // exec(opt, func, x0);

    // handy::print(x.transpose());



    // nlpp::Vec x = nlpp::Vec::Constant(10, 2.0);
    // nlpp::Vec gx(x.rows());


    // nlpp::wrap::impl::IsGradient_0<nlpp::wrap::impl::Visitor<nlpp::fd::Gradient<nlpp::Rosenbrock>>, nlpp::Vec>{};

    using Impl = nlpp::wrap::impl::Visitor<nlpp::fd::Gradient<nlpp::Rosenbrock>>;
    using V = nlpp::Vec;

    // using T = nlpp::wrap::impl::gradientType<Impl, nlpp::impl::Plain<V>, nlpp::impl::Plain<V>&>;
    // using T = nlpp::wrap::impl::funcGradType<nlpp::fd::Gradient<nlpp::Rosenbrock>, nlpp::impl::Plain<V>, nlpp::impl::Plain<V>&>;
    // using T = nlpp::wrap::impl::funcGradType<Impl, nlpp::impl::Plain<V>, nlpp::impl::Plain<V>&>;
    // using T = nlpp::wrap::impl::IsGradient_0<Impl, nlpp::Vec>;
    // auto v = nlpp::wrap::impl::GetOpId<nlpp::wrap::impl::IsGradient_0, nlpp::Vec, std::tuple<Impl>>;

    // auto v = nlpp::wrap::impl::OpId<std::tuple<Impl>, nlpp::Vec>::Gradient_0;
    // auto v = nlpp::wrap::impl::TestId<std::tuple<Impl>, nlpp::Vec>::Gradient_0;

    using T = nlpp::impl::is_detected<TestInvoke, Impl, nlpp::impl::Plain<V>, nlpp::impl::Plain<V>&>;


    // nlpp::wrap::FunctionGradient<nlpp::Rosenbrock, nlpp::fd::Gradient<nlpp::Rosenbrock>> f(nlpp::Rosenbrock{}, nlpp::fd::Gradient<nlpp::Rosenbrock>(nlpp::Rosenbrock{}));

    // nlpp::wrap::FunctionGradient<nlpp::Rosenbrock, nlpp::fd::Gradient<nlpp::Rosenbrock>>::HasOp<nlpp::Vec>::FuncGrad_0;

    // g.gradient(x, gx);

    // handy::print(gx.transpose());


    // using V = nlpp::Vec;
    // using TFs = std::tuple<nlpp::fd::Gradient<nlpp::Rosenbrock, nlpp::fd::Forward, nlpp::fd::AutoStep>>;
    // handy::print(nlpp::wrap::impl::GetOpId<nlpp::wrap::impl::IsGradient_0, V, TFs>);


    return 0;
}


