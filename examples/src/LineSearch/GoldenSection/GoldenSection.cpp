#include "LineSearch/GoldenSection/GoldenSection.h"


double func1 (double x)
{
	return (x - 3) * pow(x, 3) * pow(x - 6, 4);
}



int main ()
{
	nlpp::GoldenSection gs;

	double x = gs(func1, 0.0, 4.0);

	handy::print(x, "      ", func1(x));



	return 0;
}
