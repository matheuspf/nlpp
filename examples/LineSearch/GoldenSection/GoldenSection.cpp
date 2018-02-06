#include "GoldenSection.h"


double func1 (double x)
{
	return (x - 3) * pow(x, 3) * pow(x - 6, 4);
}



int main ()
{
	GoldenSection gs;

	double x = gs.lineSearch(func1, 0.0, 4.0);

	DB(x << "      " << func1(x));



	return 0;
}
