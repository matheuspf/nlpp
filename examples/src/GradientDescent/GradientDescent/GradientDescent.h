#include "../../../../include/nlpp/GradientDescent/GradientDescent.h"
#include "../Helpers/Optimizer.h"


namespace js_nlp
{

struct GD : public nlpp::GradientDescent<>,
            public Optimizer<GD>
{
    GD();

    using Optimizer<GD>::optimize;
};

}