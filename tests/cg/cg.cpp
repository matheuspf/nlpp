#include <catch2/catch_all.hpp>
#include <boost/mp11.hpp>

#include "cg/cg.hpp"
#include "line_search/goldstein/goldstein.hpp"
#include "TestFunctions/Rosenbrock.h"


template <class Tuple>
struct OptimizerFixture
{
    using Optimizer = std::tuple_element_t<0, Tuple>;
    using FloatType = std::tuple_element_t<1, Tuple>;

    template <class Function, class Domain, class Constraints>
    static auto optimize (const Optimizer& optimizer, const Function& function,
                          const Domain& domain, const Constraints& constraints)
    {
        auto output = optimizer.opt(function, domain, constraints);
        auto status = std::get<std::tuple_size_v<decltype(output)>-1>(output);

        bool converged = bool(status.code & (
            nlpp::Status::Code::FunctionCondition |
            nlpp::Status::Code::GradientCondition |
            nlpp::Status::Code::HessianCondition  |
            nlpp::Status::Code::NumIterations
        ));

        INFO("Status: " << status);
        INFO("Optimizer: " << nlpp::impl::type_name<Optimizer>());
        INFO("Function: " << nlpp::impl::type_name<Function>());
        INFO("Domain: " << nlpp::impl::type_name<Domain>());
        INFO("Constraints: " << nlpp::impl::type_name<Constraints>());

        CHECK(converged);

        return output;
    }
};

template <class Float>
using CGParams = boost::mp11::mp_product<nlpp::CG,
    std::tuple<nlpp::FR, nlpp::FR_PR>,
    std::tuple<::nlpp::ls::StrongWolfe<Float>, ::nlpp::ls::Goldstein<Float>>,
    std::tuple<nlpp::stop::GradientOptimizer<true, Float>>,
    std::tuple<nlpp::out::GradientOptimizer<0, Float>>
>;

using CGTypes = std::tuple<float, double, long double>;

using CGConfig = boost::mp11::mp_product<std::tuple, 
    boost::mp11::mp_flatten<
        boost::mp11::mp_transform<CGParams, CGTypes>
    >,
    CGTypes
>;


TEMPLATE_LIST_TEST_CASE_METHOD(OptimizerFixture, "CG tests", "[abc]", CGConfig)
{
    using Fixture = OptimizerFixture<TestType>;
    using Optimizer = typename Fixture::Optimizer;
    using FloatType = typename Fixture::FloatType;
    using Vec = nlpp::VecX<FloatType>;

    Optimizer optimizer;


    //auto x0 = GENERATE(as<Vec>{}, Vec::Constant(2, 2.0), Vec::Constant(5, 2.0));
    Vec x0 = Vec::Constant(5, 2.0);

    auto function = nlpp::wrap::fd::funcGrad(nlpp::Rosenbrock{});
    auto domain = nlpp::wrap::StartDomain<Vec>(x0);
    auto constraints = nlpp::wrap::Unconstrained<>{};

    Fixture::optimize(optimizer, function, domain, constraints);
}