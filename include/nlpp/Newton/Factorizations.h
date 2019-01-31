#pragma once

#include "../Helpers/Helpers.h"



namespace nlpp
{

namespace fact
{

struct SimplyInvert
{
	template <class V, class U>
	impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, const Eigen::MatrixBase<U>& hess)
	{
		return -hess.colPivHouseholderQr().solve(grad);
	}
};


template <typename Float = types::Float>
struct SmallIdentity
{
	SmallIdentity (Float alpha = 1e-5) : alpha(alpha) {}

	template <class V, class U>
	impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess)
	{
		auto minDiag = hess.diagonal().array().minCoeff();

		if(minDiag < 0.0)
			hess.diagonal().array() += minDiag + alpha;

		return -hess.colPivHouseholderQr().solve(grad);
	}

	Float alpha;
};



template <typename Float = types::Float>
struct CholeskyIdentity
{
	CholeskyIdentity (Float beta = 1e-3, Float c = 2.0, Float maxTau = 1e8) : beta(beta), c(c), maxTau(maxTau) {}

	template <class V, class U>
	impl::Plain<V> operator () (const Eigen::MatrixBase<V>& grad, U hess)
	{
		impl::PlainArray<U> orgDiag = hess.diagonal().array();

		auto minDiag = orgDiag.minCoeff();

		Float tau = minDiag < 0.0 ? beta - minDiag : 0.0;

		while(tau < maxTau)
		{
			Eigen::LLT<impl::Plain<U>> llt(hess);

			if(llt.info() == Eigen::Success)
				return -llt.solve(grad);

			hess.diagonal().array() = orgDiag + tau;

			tau = std::max(c * tau, beta);
		}

		return -grad;
	}

	Float beta;
	Float c;
	Float maxTau;
};


template <typename Float = types::Float>
struct CholeskyFactorization
{
	CholeskyFactorization (Float delta = 1e-3) : delta(delta) {}

	Vec operator () (const Vec& grad, Mat hess)
	{
		int N = hess.rows();

		double maxDiag = -1e20, maxOffDiag = -1e20;

		for(int i = 0; i < N; ++i)
		{
			maxDiag = std::max(maxDiag, std::abs(hess(i, i)));

			for(int j = 0; j < N; ++j) if(i != j)
				maxOffDiag = std::max(maxOffDiag, std::abs(hess(i, j)));
		}

		double beta = std::max(constants::eps, std::max(maxDiag, maxOffDiag / std::max(1.0, sqrt(N*N - 1.0))));


		Mat L = Mat::Identity(N, N);
		Mat C = Mat::Identity(N, N);
		Mat D = Mat::Constant(N, N, 0.0);
		Mat E = Mat::Constant(N, N, 0.0);
		Mat Q = Mat::Identity(N, N);
		Mat O = Mat::Identity(N, N);


		for(int i = 0; i < N; ++i)
			C(i, i) = hess(i, i);


		for(int i = 0; i < N; ++i)
		{
			double val = -1e20;
			int p = 0;

			for(int j = i; j < N; ++j) if(std::abs(hess(j, j)) > val)
				val = std::abs(hess(j, j)), p = j;

			if(p != i)
			{
				Mat P = Mat::Identity(N, N);

				P(i, i) = P(p, p) = 0.0;
				P(i, p) = P(p, i) = 1.0;

				hess = P * hess * P.transpose();

				Q = P * Q;
				O = O * P.transpose();
			}



			double phi = -1e20;

			for(int j = 0; j < i; ++j)
				L(i, j) = C(i, j) / D(j, j);

			for(int j = i+1; j < N; ++j)
			{
				C(j, i) = hess(j, i);

				for(int k = 0; k < i; ++k)
					C(j, i) -= L(i, k) * C(j, k);
				
				phi = std::max(phi, std::abs(C(j, i)));
			}

			if(i == N-1)
				phi = 0.0;

			D(i, i) = std::max(delta, std::max(std::abs(C(i, i)), pow(phi, 2) / beta));

			E(i, i) = D(i, i) - C(i, i);


			for(int j = i+1; j < N; ++j)
				C(j, j) = C(j, j) - pow(C(j, i), 2) / D(i, i);
		}


		hess = Q.inverse() * (hess + E) * O.inverse();

		return -hess.colPivHouseholderQr().solve(grad);
	}


	double delta;
};




struct IndefiniteFactorization
{
	IndefiniteFactorization (double delta = 1e-2) : delta(delta) {}


	Vec operator () (const Vec& grad, Mat hess)
	{
		int N = hess.rows();


		Eigen::RealSchur<Mat> schur(hess);

		Eigen::EigenSolver<Mat> eigen(schur.matrixT());

		const Vec& eigVal = eigen.eigenvalues().real();

		const Mat& eigVec = eigen.eigenvectors().real();


		Mat F = Mat::Constant(N, N, 0.0);

		for(int i = 0; i < N; ++i)
			F(i, i) = (eigVal[i] < delta ? delta - eigVal[i] : 0.0);

		F = schur.matrixT() + eigVec * F * eigVec.transpose();


		return -schur.matrixU() * F.llt().solve(Mat::Identity(N, N)) * schur.matrixU().transpose() * grad;
	}

	double delta;
};

} // namespace fact

} // namespace nlpp
