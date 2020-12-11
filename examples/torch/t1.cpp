#include <torch/torch.h>
#include <iostream>



auto func (const torch::Tensor& x)
{
    torch::AutoGradMode guard(true);
    return x.norm();
}

auto grad (torch::Tensor& x, torch::Tensor& fx)
{
    fx.backward();
    auto g = x.grad().clone();
    x.grad().zero_();

    return g;
}

int main()
{
    torch::Device device = torch::kCUDA;
    // torch::Tensor x = torch::ones(2, torch::requires_grad()).cuda();
    torch::Tensor x = torch::ones(2, device);
    x[1] = 2.0;
    x.set_requires_grad(true);

    torch::AutoGradMode guard(false);

    for(int i = 0; i < 10; ++i)
    {
        auto fx = func(x);
        auto gx = grad(x, fx);

        std::cout << x << "\n";

        // torch::NoGradGuard guard;
        x -= 1e-1 * gx;
    }

    std::cout << x << "\n\n";

    return 0;
}