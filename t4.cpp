#include <type_traits>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>


using namespace std;



int main ()
{
    Eigen::VectorXd x(10);
    std::cout << x.size() << "\n";

    return 0;
}
