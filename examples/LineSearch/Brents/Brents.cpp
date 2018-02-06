#include "LineSearch/Brents/Brents.h"


double func1 (double x)
{
	return (x - 3) * std::pow(x, 3) * std::pow(x - 6, 4);
}



int main ()
{
	cppnl::Brents br;

	double x = br(func1, 0.0, 7.0);

	handy::print(x, "     ", func1(x));



	return 0;
}
