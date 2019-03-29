#pragma once

#include "Helpers.h"
#include <Spectra/SymEigsSolver.h>


namespace nlpp
{

/// Simple interface to select some of the eigenvalues and eigenvectors of a Eigen::MatrixBase
template <Spectra::SELECT_EIGENVALUE Rule = Spectra::SELECT_EIGENVALUE::LARGEST_ALGE>
struct TopEigen
{
    template <class Mat>
    TopEigen (const Eigen::DenseBase<Mat>& X, int K) : operation(X), solver(&operation, K, std::min(int(X.rows()), 2*K))
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

} // namespace nlpp