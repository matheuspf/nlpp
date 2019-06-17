// #include "Helpers/Output.h"
#include "Newton/Newton.h"


int main ()
{
    nlpp::Newton<> opt;

    nlpp::Vec x = nlpp::Vec::Constant(5, 1.2);


    // nlpp::out::Optimizer<1> out;

    // out(opt, x, 0.0);


    return 0;
}