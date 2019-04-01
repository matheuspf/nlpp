#include <benchmark/benchmark.h>

#include "TrustRegion/IterativeTR/IterativeTR.h"
#include "TestFunctions/Rosenbrock.h"


template <class Function>
static void BM_iterativeTR (benchmark::State& state, Function func)
{
    nlpp::IterativeTR<> opt;

    nlpp::Vec x0 = nlpp::Vec::Constant(state.range(0), 5.0);

    for(auto _ : state)
        nlpp::Vec x = opt(func, x0);
}


BENCHMARK_CAPTURE(BM_iterativeTR, rosenbrock, nlpp::Rosenbrock{})->Range(10, 100);