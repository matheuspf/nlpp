#include "citgo/knn/kdtree.hpp"
#include "citgo/knn/default.hpp"

using namespace nlpp;


Matu naiveKNN (const Mat& X, int k)
{
    Matu knn(k+1, X.cols());
    std::vector<std::pair<double, std::size_t>> dist(X.cols());

    for(int i = 0; i < X.cols(); ++i)
    {
        for(int j = 0; j < X.cols(); ++j)
            dist[j] = {(X.col(i) - X.col(j)).norm(), j};

        std::sort(dist.begin(), dist.end());

        for(int j = 0; j < k+1; ++j)
            knn(j, i) = dist[j].second;
    }

    return knn;
}

int main ()
{
    constexpr int N = 5;
    constexpr int K = 3;

    Mat X = Mat::Random(N, 2*N);

    // std::cout << X.rows() << "\t" << X.cols() << "\n";

    KDTreeKNN kdtree;
    DefaultKNN deft;

    auto knn_naive = naiveKNN(X, K);
    auto knn_kdtree = kdtree(X, K);
    auto knn_default = deft(X, K);

    std::cout << knn_naive << "\n\n";
    std::cout << knn_kdtree << "\n\n";
    std::cout << knn_default << "\n";



    return 0;
}
