#include "LineSearch/Bracketing/Bracketing.hpp"


double func1 (double x)
{
	return (x - 3) * pow(x, 3) * pow(x - 6, 4);
}



int main ()
{
	nlpp::Bracketing bc;

	auto [x, y, z] = bc(func1, 0, 3);

	handy::print(x, "  ", y, "  ", z);


	return 0;
}