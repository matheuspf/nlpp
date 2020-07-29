#pragma once

#include "helpers/helpers.hpp"

#include <coin/IpTNLP.hpp>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>

using namespace Ipopt;

namespace nlpp
{

template <class FunctionType>
struct LocalSearch
{
    using Function = FunctionType;
    using Vector = typename Function::Vector;

    LocalSearch(const Function &) {}

    Vector operator()(Function &function, Vector x, const Vector &lowerBound, const Vector &upperBound, int iter = 1e2, int prt = 0)
    {
        SmartPtr<TNLP> mynlp = new MyNLP<Function, Vector>();

        ((MyNLP<Function, Vector> &)*mynlp).start = x;

        ((MyNLP<Function, Vector> &)*mynlp).func = &function;

        SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

        app->Options()->SetStringValue("jacobian_approximation", "finite-difference-values");
        app->Options()->SetStringValue("hessian_approximation", "limited-memory");

        //app->Options()->SetIntegerValue("limited_memory_max_history", 20);

        // app->Options()->SetStringValue("warm_start_init_point", "yes");
        // app->Options()->SetNumericValue("warm_start_bound_push", 1e-6);
        // app->Options()->SetNumericValue("warm_start_mult_bound_push", 1e-6);
        //app->Options()->SetNumericValue("mu_init", 1e-2);
        //app->Options()->SetStringValue("nlp_scaling_method", "none");

        // app->Options()->SetNumericValue("compl_inf_tol", 1e-9);

        app->Options()->SetNumericValue("tol", 1e-16);
        app->Options()->SetNumericValue("dual_inf_tol", 1e-16);
        app->Options()->SetNumericValue("acceptable_tol", 1e-16);
        app->Options()->SetNumericValue("acceptable_dual_inf_tol", 1e-16);
        app->Options()->SetNumericValue("constr_viol_tol", 1e-20);
        app->Options()->SetNumericValue("acceptable_constr_viol_tol", 1e-20);
        //app->Options()->SetIntegerValue("print_level", prt ? 3 : 0);
        app->Options()->SetIntegerValue("print_level", 0);
        app->Options()->SetIntegerValue("max_iter", iter);

        //app->Options()->SetStringValue("start_with_resto", "yes");

        // app->Options()->SetNumericValue("tol", 1e-3);
        // app->Options()->SetNumericValue("acceptable_tol", 3e-3);
        // app->Options()->SetStringValue("expect_infeasible_problem", "yes");
        // app->Options()->SetNumericValue("dual_inf_tol", 1e-6);
        // app->Options()->SetNumericValue("compl_inf_tol", 1e-6);
        // app->Options()->SetNumericValue("acceptable_dual_inf_tol", 3e-3);
        // app->Options()->SetNumericValue("acceptable_compl_inf_tol", 3e-3);
        // app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
        // app->Options()->SetStringValue("alpha_for_y", "full");

        // app->Options()->SetStringValue("warm_start_init_point", "yes");
        // app->Options()->SetNumericValue("warm_start_bound_push", 1e-9);
        // app->Options()->SetNumericValue("warm_start_bound_frac", 1e-9);
        // app->Options()->SetNumericValue("warm_start_slack_bound_frac", 1e-9);
        // app->Options()->SetNumericValue("warm_start_slack_bound_push", 1e-9);
        // app->Options()->SetNumericValue("warm_start_mult_bound_push", 1e-9);

        ApplicationReturnStatus status;
        status = app->Initialize();

        // if (status != Solve_Succeeded)
        // {
        // 	std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        // 	exit(0);
        // }

        status = app->OptimizeTNLP(mynlp);

        // if (status == Solve_Succeeded) {
        // // Retrieve some statistics about the solve
        // Index iter_count = app->Statistics()->IterationCount();
        // std::cout << std::endl << std::endl << "*** The problem solved in " << iter_count << " iterations!" << std::endl;

        // Number final_obj = app->Statistics()->FinalObjective();
        // std::cout << std::endl << std::endl << "*** The final value of the objective function is " << final_obj << '.' << std::endl;
        // }

        x = ((MyNLP<Function, Vector> &)*mynlp).start;

        // for(int i = 0; i < x.size(); ++i)
        // 	x[i] = std::min(function.upperBound[i], std::max(function.lowerBound[i], x[i]));

        function(x);

        return x;
    }

    template <class Function, class Vector>
    struct MyNLP : TNLP
    {
        MyNLP(int numEqualities, int numInequalities) : numEqualities(numEqualities), numInequalities(numInequalities) {}

        bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g,
                          Index &nnz_h_lag, IndexStyleEnum &index_style)
        {
            n = func->N;

            //m = 1;
            m = numEqualities + numInequalities;

            nnz_jac_g = n * m;
            //nnz_jac_g = 16;

            nnz_h_lag = (n * (n + 1)) / 2;

            index_style = FORTRAN_STYLE;

            return true;
        }

        bool get_bounds_info(Index n, Number *x_l, Number *x_u,
                             Index m, Number *g_l, Number *g_u)
        {
            // std::copy(func->lowerBound.begin(), func->lowerBound.end(), x_l);
            // std::copy(func->upperBound.begin(), func->upperBound.end(), x_u);

            for (int i = 0; i < n; ++i)
                x_l[i] = func->lowerBound[i], x_u[i] = func->upperBound[i];
            //x_l[i] = func->lowerBound[i] - 1e-10, x_u[i] = func->upperBound[i] + 1e-10;

            // std::fill(g_l, g_l + FunctionType::numEqualities, -1e19);
            // std::fill(g_u, g_u + FunctionType::numEqualities, 0.0);

            // std::fill(g_l + FunctionType::numEqualities, g_l + m, -1e19);
            // std::fill(g_u + FunctionType::numEqualities, g_u + m, 0.0);

            std::fill(g_l, g_l + m, -1e19);
            std::fill(g_u, g_u + m, 0.0);

            //g_l[0] = g_u[0] = 0.0;

            //g_l[m] = g_u[m] = 0.0;
            //g_l[m] = -1e19; g_u[m] = 0.0;

            return true;
        }

        bool get_starting_point(Index n, bool init_x, Number *x,
                                bool init_z, Number *z_L, Number *z_U,
                                Index m, bool init_lambda,
                                Number *lambda)
        {
            std::copy(start.begin(), start.end(), x);

            return true;
        }

        double function(Index n, const Number *x)
        {
            std::copy(x, x + n, aux.begin());

            return (*func)(aux);
        }

        bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
        {
            obj_value = function(n, x);
            //obj_value = 0.0;

            //obj_value = std::pow(function(n, x) - func->optimal, 2);

            return true;
        }

        void finiteDifference(Index n, const Number *x, Number *grad_f, double h = 1e-8)
        {
            Number y[n];

            std::copy(x, x + n, &y[0]);

            for (int i = 0; i < n; ++i)
            {
                y[i] = x[i] - h;

                double g1 = function(n, y);

                y[i] = x[i] + h;

                double g2 = function(n, y);

                grad_f[i] = (g2 - g1) / (2 * h);

                y[i] = x[i];
            }

            // double fx = function(n, y);

            // for(int i = 0; i < n; ++i)
            // {
            // 	y[i] = x[i] + h;

            // 	double gx = function(n, y);

            // 	grad_f[i] = (gx - fx) / h;

            // 	y[i] = x[i];
            // }
        }

        bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
        {
            finiteDifference(n, x, grad_f);

            return true;
        }

        void constraints(Vector &x, double *g)
        {
            double f = 0;

            func->func(&x[0], &f, &func->ineqs[0], &func->eqs[0], int(x.size()));

            // g[0] = 0.0;

            // for(double y : Base::ineqs)
            //     g[0] += std::max(0.0, y);

            // for(double y : Base::eqs)
            //     g[0] += (std::abs(y) <= 1e-4 ? 0.0 : std::abs(y));

            // g[0] += std::abs(f - Base::optimal);

            for (int i = 0; i < Function::numEqualities; ++i)
                g[i] = std::sqrt(std::abs(func->eqs[i])) - 1e-2;
            //g[i] = std::abs(Base::eqs[i]) <= Base::eps ? std::pow(Base::eqs[i], 2) : std::abs(Base::eqs[i]);
            //g[i] = std::abs(Base::eqs[i]) <= 1e-4 ? 0.0 : Base::eqs[i];
            //g[i] = Base::eqs[i];
            //g[i] = (std::pow(func->eqs[i], 2) - 1e-8);
            //g[i] = std::abs(func->eqs[i]) - 9.99e-5;
            //g[i] = (std::abs(func->eqs[i]) - 1e-4);
            //g[i] = std::abs(func->eqs[i]);

            for (int i = 0; i < Function::numInequalities; ++i)
                //g[i + Base::numEqualities] = std::max(0.0, Base::ineqs[i]);
                g[i + Function::numEqualities] = func->ineqs[i];
            //g[i + Base::numEqualities] = (Base::ineqs[i] < 1e-10 ? 1e8 : 1e5) * Base::ineqs[i];

            //g[Function::numEqualities + Function::numInequalities] = std::pow(f - func->optimal, 2) - 1e-8;
            // std::pow(f - Base::optimal, 4) : std::abs(f - Base::optimal);

            //g[Function::numEqualities + Function::numInequalities] = 1e7 * std::pow(f - func->optimal, 2);
        }

        bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
        {
            std::copy(x, x + n, aux.begin());

            constraints(aux, g);

            return true;
        }

        bool eval_jac_g(Index n, const Number *x, bool new_x,
                        Index m, Index nele_jac, Index *iRow, Index *jCol,
                        Number *values)
        {
            if (values == NULL)
            {
                for (int i = 0, k = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j, ++k)
                        iRow[k] = i + 1, jCol[k] = j + 1;
            }

            // if(values == NULL)
            // {
            // 	iRow[0] = 1, jCol[0] = 1;
            // 	iRow[1] = 1, jCol[1] = 2;
            // 	iRow[2] = 1, jCol[2] = 3;
            // 	iRow[3] = 2, jCol[3] = 3;
            // 	iRow[4] = 2, jCol[4] = 4;
            // 	iRow[5] = 2, jCol[5] = 5;
            // 	iRow[6] = 2, jCol[6] = 6;
            // 	iRow[7] = 3, jCol[7] = 2;
            // 	iRow[8] = 3, jCol[8] = 4;
            // 	iRow[9] = 3, jCol[9] = 7;
            // 	iRow[10] = 4, jCol[10] = 4;
            // 	iRow[11] = 4, jCol[11] = 5;
            // 	iRow[12] = 5, jCol[12] = 4;
            // 	iRow[13] = 5, jCol[13] = 6;
            // 	iRow[14] = 6, jCol[14] = 4;
            // 	iRow[15] = 6, jCol[15] = 7;
            // }

            // else
            // {
            // 	values[0] = -1.0;
            // 	values[1] = 35*0.6*std::pow(x[1], -0.4);
            // 	values[2] = 35*0.6*std::pow(x[2], -0.4);
            // 	values[3] = x[3] - 300.0;
            // 	values[4] = x[2] - 25*x[4] + 25*x[5];
            // 	values[5] = -25*x[3] + 7500.0;
            // 	values[6] = 25*x[3] - 7500.0;
            // 	values[7] = -x[3] + 100.0;
            // 	values[8] = 155.365 - x[1] - 25*x[6];
            // 	values[9] = 2500.0 - 25*x[3];
            // 	values[10] = -1.0/(-x[3] + 900.0);
            // 	values[11] = -1.0;
            // 	values[12] = 1.0/(x[3] + 300.0);
            // 	values[13] = -1.0;
            // 	values[14] = -2.0/(-2*x[3] + 700.0);
            // 	values[15] = -1.0;
            // }

            return true;
        }

        bool eval_h(Index n, const Number *x, bool new_x,
                    Number obj_factor, Index m, const Number *lambda,
                    bool new_lambda, Index nele_hess, Index *iRow,
                    Index *jCol, Number *values)
        {
            return true;
        }

        void finalize_solution(SolverReturn status,
                               Index n, const Number *x, const Number *z_L, const Number *z_U,
                               Index m, const Number *g, const Number *lambda,
                               Number obj_value,
                               const IpoptData *ip_data,
                               IpoptCalculatedQuantities *ip_cq)
        {
            std::copy(x, x + n, start.begin());
        }

        Vector start;
        Function *func;

        Vector aux;

        int numEqualities;
        int numInequalities;
    };
};

} // namespace nlpp
