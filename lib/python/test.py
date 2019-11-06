import numpy as np
import nlpy

cg = nlpy.CG(delta=1e-4, stop=nlpy.stop.GradientOptimizer(1), out=nlpy.out.GradientOptimizer(1))
# cg.ls = nlpy.ls.Goldstein(1e-4)

func = lambda x: (x - 0.5).norm()

x0 = np.array([2.0, 0.0])

x = cg(func, x0)

print(x)