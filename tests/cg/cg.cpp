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

    template <class... Args>
    static auto optimize (const Optimizer& optimizer, Args&&... args)
    {
        auto output = optimizer(std::forward<Args>(args)...);
        auto status = std::get<std::tuple_size_v<decltype(output)>-1>(output);

        auto converged = nlpp::Status(
            status.code &
            (nlpp::Status::Code::FunctionCondition |
             nlpp::Status::Code::GradientCondition |
             nlpp::Status::Code::HessianCondition  |
             nlpp::Status::Code::NumIterations)
        );

        INFO(converged);
        CHECK(!converged);

        return output;
    }
};

template <class Float>
using CGParams = boost::mp11::mp_product<nlpp::CG,
    std::tuple<nlpp::FR, nlpp::FR_PR>,
    std::tuple<::nlpp::ls::StrongWolfe<Float>, ::nlpp::ls::Goldstein<Float>>
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

    nlpp::Rosenbrock func;

    // auto x0 = GENERATE(as<Vec>{}, Vec::Constant(5, 2.0), Vec::Constant(5, 2.0));
    Vec x0 = Vec::Constant(5, 2.0);

    Fixture::optimize(optimizer, func, x0);
}