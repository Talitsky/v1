---
layout: post
title: "Kernel Functions"
# author: "AT"
categories: documentation
tags: [documentation]
image: documents-control-plan-kernel.png
---

## Kernel Functions 
Firstly, we realized the *KernelFunctions* to compute TK matrices and a monomial basis, given 
$$x$$
and degree 
$$q$$
one  can call

```python
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

```python
from PMKLpy import Transformation
Z = Transformation.monomials(x, 1)
```

To compute TK, the user need to call *makeK* function 

```python
TK = KernelFunctions.makeK(Kernel, P) 
```
*makeK* is a function that returns a Kernel matrix *(n\_samples, n\_samples)*. It requires a *P* matrix of the form *(2q, 2q)* and an object Kernel. 

For general Kernel matrices *K(X1, X2)* we realized a function *TKtest*. This function takes two matrices of inputs, as well as monomial basis of the inputs  *Z1, Z2* and a lower  *a* and upper  *b* bound over which we integrate  and a matrix  *P* that parameterizes the TK kernel function.  Computes the test kernel matrix for a TK kernel.

```python
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
