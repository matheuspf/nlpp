#include "Brents.h"


double func1 (double x)
{
	return (x - 3) * pow(x, 3) * pow(x - 6, 4);
}



int main ()
{
	Brents br;

	double x = br.lineSearch(func1, 0.0, 7.0);

	DB(x << "      " << func1(x));



	return 0;
}
