#include <nlpp/CG/CG.h>
#include "Helpers/Optimizer.h"


EMSCRIPTEN_BINDINGS(js_nlp)
{

    emscripten::class_<nlpp::poly::GradientOptimizer<>>("GradientOptimizer");

    NLPP_JS_OPTIMIZER(CG);
}