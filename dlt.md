# Mathematical Foundations of Deep Learning at RWTH Aachen (Winter 2023-24)

Instructor: Semih Cayci (cayci[at]mathc.rwth-aachen.de)

## 0. Basics of Machine Learning

1. [Concentration inequalities: Chernoff-Hoeffding](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/2_Concentration%20Inequalities%20for%20Machine%20Learning/1_ChernoffHoeffding.pdf)
2. [Concentration inequalities: Martingale-based bounds](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/2_Concentration%20Inequalities%20for%20Machine%20Learning/2_Azuma-McDiarmid.pdf)
3. [Basic supervised learning](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/3_Basics%20of%20(Supervised)%20Learning%20Theory/1_Basic%20Supervised%20Learning.pdf)
4. [Empirical risk minimization](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/3_Basics%20of%20(Supervised)%20Learning%20Theory/2_ERM.pdf)

[Exercise sheet 0: Concentration inequalities](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/voluntary-exercise-sheet0.pdf)

[Exercise sheet 1: Basics of PAC-SL](https://github.com/semihcayci/deeplearningtheory/blob/17170abf7c283274caf01e2abf2629e0b55d2c1a/Assignment1.pdf)

## 1. Optimization for Deep Learning

### Basics of Convex Optimization
1. [ERM as an optimization problem](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/1_ERM%20as%20Optimization.pdf)
2. [Convex optimization basics](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/2_Convex%20Optimization%20Basics.pdf)
3. [Projected subgradient descent](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/3_Projected%20Subgradient%20Descent.pdf)
4. [Gradient descent for strongly-convex and non-smooth optimization](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/4_Gradient%20Descent%20for%20Strongly___vex%20Nonsmooth%20Optimization.pdf)
5. [Gradient descent for smooth optimization](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/5_Gradient%20Descent%20for%20Smooth%20a___trongly%20Convex%20Functions%202.pdf)
6. [Stochastic gradient descent (SGD)](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/6_SGD%204.pdf)
### Gradient descent for overparameterized ReLU networks -- NTK-Based Approach
1. [Main idea: Linearization around random initialization](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/7_Optimization%20in%20Neural%20Tangent%20Kernel%20Regime/1.%20Linearization.pdf)
2. [Neural tangent kernel](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/7_Optimization%20in%20Neural%20Tangent%20Kernel%20Regime/2.%20Neural%20Tangent%20Kernel%20-%20Slides.pdf)
3. [Gradient descent for regression](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/7_Optimization%20in%20Neural%20Tangent%20Kernel%20Regime/3a.%20Gradient%20Descent%20-%20Regression.pdf)
4. [Supplementary: a study of linear regression](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/7_Optimization%20in%20Neural%20Tangent%20Kernel%20Regime/3b.%20Gradient%20Descent%20for%20Linear%20Regression%20-%20Slides.pdf)
5. [Massive overparameterization: Convergence of gradient flow](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/4_Optimization%20for%20Deep%20Learning/7_Optimization%20in%20Neural%20Tangent%20Kernel%20Regime/4.%20Gradient%20Flow%20under%20Overparameterization.pdf)

[Exercise sheet 2: Convex optimization algorithms](https://github.com/semihcayci/deeplearningtheory/blob/f99b7bc29671d3ea0fc0ebe3e67906e17e9f41eb/Assignment2.pdf)

[Exercise sheet 3: Analysis of GD in the NTK regime](https://github.com/semihcayci/deeplearningtheory/blob/c42abce4a47a1e0e73e7d04022b69f230416c17c/Assignment3.pdf)
   

This book is on the non-asymptotic theory of high-dimensional statistics. It starts with an elegant treatment of tail and concentration bounds, and covers modern statistical subjects: non-parametric regression, RKHS (reproducible kernel Hilbert spaces), random matrix theory, graphical models, sparse linear models, uniform laws, etc. 

For _concentration bounds_ and _kernel methods_, this book is my favorite.

## Generalization in Deep Learning
### [A. Topics in Non-Parametric Statistics - A. Nemirovski](https://www2.isye.gatech.edu/~nemirovs/Lect_SaintFlour.pdf)

How can you perform statistical inference on infinite-dimensional parameters (e.g., functions, time-dependent signals) from noisy observations? This book provides concise answers to this question by focusing on estimating non-parametric regression to functions and functionals. It is a bit difficult to read because of the notation, but it seems to be highly insightful and self-contained.

### [B. Statistical Inference via Convex Optimization - A. Juditsky and A. Nemirovski](https://www2.isye.gatech.edu/~nemirovs/StatOptNoSolutions.pdf)

It is a brand new book due April 2020 on statistical inference. The subjects include (2A) and extend to signal recovery, (sequential) hypothesis testing and sparse recovery.

## Universal Approximation Theorems for Deep Learning

Classical book on large sample properties and approximations of statistical tests, estimators and procedures.
