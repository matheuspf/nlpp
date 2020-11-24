#include <catch2/catch_all.hpp>
#include <boost/mp11.hpp>

#include "line_search/goldstein/goldstein.hpp"
#include "TestFunctions/Rosenbrock.h"

#include "cg/cg.hpp"
#include "newton/newton.hpp"


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
            nlpp::Status::Code::VariableCondition |
            nlpp::Status::Code::FunctionCondition |
            nlpp::Status::Code::GradientCondition |
            nlpp::Status::Code::HessianCondition
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


using OptimizerTypes = std::tuple<float, double, long double>;

template <template <class Float> class OptimizerParams>
using OptimizerConfig = boost::mp11::mp_product<std::tuple, 
    boost::mp11::mp_flatten<
        boost::mp11::mp_transform<OptimizerParams, OptimizerTypes>
    >,
    OptimizerTypes
>;

template <class Float>
using CGParams = boost::mp11::mp_product<nlpp::CG,
    std::tuple<nlpp::FR, nlpp::FR_PR>,
    std::tuple<::nlpp::ls::StrongWolfe<Float>, ::nlpp::ls::Goldstein<Float>>,
    std::tuple<nlpp::stop::GradientOptimizer<false, Float>>,
    std::tuple<nlpp::out::GradientOptimizer<0, Float>>
>;

template <class Float>
using NewtonParams = boost::mp11::mp_product<nlpp::Newton,
    std::tuple<nlpp::fact::SmallIdentity<Float>, nlpp::fact::CholeskyIdentity<Float>,
               nlpp::fact::CholeskyFactorization<Float>, nlpp::fact::IndefiniteFactorization<Float>>,
    std::tuple<::nlpp::ls::StrongWolfe<Float>, ::nlpp::ls::Goldstein<Float>>,
    std::tuple<nlpp::stop::GradientOptimizer<false, Float>>,
    std::tuple<nlpp::out::GradientOptimizer<0, Float>>
>;

using CGConfig = OptimizerConfig<CGParams>;
using NewtonConfig = OptimizerConfig<CGParams>;

//using Config = std::tuple<boost::mp11::mp_flatten<CGConfig>, boost::mp11::mp_flatten<NewtonConfig>>;
using Config = decltype(std::tuple_cat(std::declval<CGConfig>(), std::declval<NewtonConfig>()));


TEMPLATE_LIST_TEST_CASE_METHOD(OptimizerFixture, "Optimizer tests", "[Optimizer]", Config)
{
    using Fixture = OptimizerFixture<TestType>;
    using Optimizer = typename Fixture::Optimizer;
    using FloatType = typename Fixture::FloatType;

    Optimizer optimizer;

    SECTION("Rosenbrock")
    {
        auto function = nlpp::wrap::fd::funcGrad(nlpp::Rosenbrock{});
        auto constraints = nlpp::wrap::Unconstrained<>{};

        SECTION("Dynamic")
        {
            auto domain = nlpp::wrap::startDomain(nlpp::VecX<FloatType>::Constant(5, 2.0));
            Fixture::optimize(optimizer, function, domain, constraints);
        }

        SECTION("Static")
        {
            auto domain = nlpp::wrap::startDomain(Eigen::Matrix<FloatType, 5, 1>::Constant(2.0));
            Fixture::optimize(optimizer, function, domain, constraints);
        }
    }
}