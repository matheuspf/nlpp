#include "benchmark/benchmark.h"

#include "Helpers/FiniteDifference.h"
#include "TestFunctions/Rosenbrock.h"


template <class Function>
static void BM_finiteGradient (benchmark::State& state, Function func)
{
    auto grad = nlpp::fd::gradient(func);

    nlpp::Vec x(state.range(0));
    std::for_each(x.data(), x.data() + x.size(), [](auto& xi){ xi = handy::rand(-10.0, 10.0); });

    for(auto _ : state)
        auto g = grad(x);
}


BENCHMARK_CAPTURE(BM_finiteGradient, rosenbrock, nlpp::Rosenbrock{})->Range(10, 100)->Complexity();


BENCHMARK_MAIN();