---
layout: post
title: "Kernel Functions"
author: "AT"
categories: documentation
tags: [documentation]
image: city-1.jpg
---

## Kernel Functions 
Firstly, we realized the *KernelFunctions* to compute TK matrices and a monomial basis, given $x$ and degree $q$ one  can call

```
import numpy as np
from PMKLpy import KernelFunctions
x = np.random.rand(100, 1)
q = 1
Lower, Upper =  x.min(axis = 0) - 0.1 , x.max(axis = 0) + 0.1
Kernel = KernelFunctions.Kernel(x, Lower, Upper,  q)
```

*Kernel* is a special class for fast computing of TK. The first argument of is data points, the second and third are low and upper bounds for samples and the last one is degree of monomial basis.  This object has 2 attributes. The first one is *Kernel.Z*, that is a vector of monomial basis. It stores *numpy.array* object with the shape *(n\_samples, q)* where q
The second attribute is *Kernel.K* is a interior structure for fast computing of TK. 

If you need only monomial basis, then you can use the *monomials(x,d)* function takes a matrix of inputs  *x* and computes the monomial basis *Z* of degree *d* 

```
from PMKLpy import Transformation
Z = Transformation.monomials(x, 1)
```

To compute TK, the user need to call *makeK* function 

```
TK = KernelFunctions.makeK(Kernel, P) 
```
*makeK* is a function that returns a Kernel matrix *(n\_samples, n\_samples)*. It requires a *P* matrix of the form *(2q, 2q)* and an object Kernel. 

For general Kernel matrices *K(X1, X2)* we realized a function *TKtest*. This function takes two matrices of inputs, as well as monomial basis of the inputs  *Z1, Z2* and a lower  *a* and upper  *b* bound over which we integrate  and a matrix  *P* that parameterizes the TK kernel function.  Computes the test kernel matrix for a TK kernel.

```
from PMKLpy import Transformation
from PMKLpy import KernelFunctions
x1 = x[:100]
x2 = x[100:]
Lower, Upper =  x.min(axis = 0) - 0.1 , x.max(axis = 0) + 0.1
P = np.eye(6)
Z1 = Transformation.monomials(x1, 1)
Z2 = Transformation.monomials(x2, 1)

TK = KernelFunctions.TKtest(x1,x2,Z1,Z2,Lower, Upper,P)
```

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

```
from PMKLpy import PMKL
SVM = PMKL.PMKL()
SVM.fit(x, y)
ypred = SVM.predict(x)
```
