#pragma once

#include "Helpers.h"
#include <spectra/SymEigsSolver.h>


namespace cppnlp
{

template <Spectra::SELECT_EIGENVALUE Rule = Spectra::SELECT_EIGENVALUE::LARGEST_ALGE>
struct TopEigen
{
    TopEigen (const Mat& X, int K) : operation(X), solver(&operation, K, std::min(int(X.rows()), 2*K))
    {
        solver.init();
        solver.compute();
    }

    Vec eigenvalues () const
    {
        return solver.eigenvalues();
    }

    Mat eigenvectors () const
    {
        return solver.eigenvectors();
    }


    Spectra::DenseSymMatProd<double> operation;

    Spectra::SymEigsSolver<double, Rule, Spectra::DenseSymMatProd<double>> solver;
};

} // namespace cppnlp