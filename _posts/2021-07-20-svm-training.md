---
layout: post
title: "How to use PMKL"
# author: "AT"
categories: documentation
tags: [documentation]
image: documentation,jpg.jpg
---

## SVM and optimization

The main class of SVM is *PMKL*. It requires several parameters for Kernel computing and *C*-SVC or *e*-SVR.

`class PMKLpy.PMKL.PMKL(C = 10,  degree = 1, bound = 0.1, epsilon = 0.1, maxit = 100, tol = 1.e-6, probability = False, to_print= True)`
**Parameters:**
* *C, float, default=10* \\Regularization parameter, smaller values lead to mappings that are more general.
* *degree, int, default=1* \\
Degree of the TK Kernel function
* *bound, float, default=0.1*  \\Area of integration is [-*bound*,1+*bound*]
* *epsilon, float, default=0.1* Epsilon parameter for regression problems
* *maxit, float default=100* \\Maximum number of iterations. Each Iteration includes the SVM training
* *tol, float, default=1.e-6* \\Tolerance of the TKL optimization algorithm
* *probability, boolean, default= False*\\
If True the algorithm are able to predict probability for binary classification. If False the standard SVM problem will be solved
* *to\_print, boolean, default=True*    \\
If True the the algorithm will be print the objective function for each iteration, else there will not be any outputs

**Attributes**:
* *fit(X, y)* Fit the SVM model according to the given training data.
* *predict(X)* Perform classification or regression on samples in X.
* *predict_proba* Compute probabilities of possible outcomes for samples in X (only for classification). *IT IS NOT IMPLEMENTED*
* *get_params* *IT IS NOT IMPLEMENTED*

```python
from PMKLpy import PMKL
SVM = PMKL.PMKL()
SVM.fit(x, y)
ypred = SVM.predict(x)
```
