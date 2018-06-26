#include "Newton/Newton.h"

#include "TestFunctions/Rosenbrock.h"


using namespace nlpp;



// int main ()
// {
// 	using Fact = fact::CholeskyIdentity;
// 	using LS = StrongWolfe;


// 	params::Newton<Fact, LS> prs;

// 	//prs.lineSearch = LS(1.0, 0.2);


// 	Newton<Fact, LS> newton(prs);


// 	Vec x = Vec::Constant(50, 5.0);
// 	// Eigen::Matrix<double, 50, 1> x;
// 	// std::fill(&x(0), &x(0) + x.size(), 5.0);


// 	Rosenbrock func;

// 	handy::print(handy::benchmark([&]{
// 		x = newton(func, fd::gradient(func), x, fd::hessian(func));
// 	}), "\n\n");


// 	handy::print(x.transpose());




// 	return 0;
// }


int main ()
{
    using Opt = nlpp::Newton<nlpp::fact::CholeskyIdentity, nlpp::StrongWolfe,
                            nlpp::stop::GradientOptimizer<1>, nlpp::out::GradientOptimizer<>>;
                            

    typename Opt::Params params;

    params.stop.xTol = 1e-4;
    params.stop.fTol = 1e-4;
    params.stop.gTol = 1e-4;


    int N = 10;

    double x0 = 1.2;


    nlpp::Rosenbrock func;

    auto grad = nlpp::fd::gradient(func);

    auto hess = nlpp::fd::hessian(func);


    nlpp::Vec xOpt = nlpp::Vec::Constant(N, x0);

    double fOpt = 0.0;


    Opt newton(params);

    nlpp::Vec x = nlpp::Vec::Constant(N, x0);


	x = newton(func, nlpp::fd::gradient(func), x, nlpp::fd::hessian(func));


    handy::print("x: ", x.transpose(), "\nfx: ", func(x), "\ngx: ", grad(x).transpose(), "\n");





	return 0;
}