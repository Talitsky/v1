---
layout: category
title: Documentation
category: documentation
permalink: /documentation
---
## Kernel Functions 
Firstly, we realized the *KernelFunctions* to compute TK matrices and a monomial basis, given $x$ and degree $q$ one  can call
`Kernel = KernelFunctions.Kernel(x, Lower, Upper,  q)`

*Kernel* is a special class for fast computing of TK. The first argument of is data points, the second and third are low and upper bounds for samples and the last one is degree of monomial basis.  This object has 2 attributes. The first one is *Kernel.Z*, that is a vector of monomial basis. It stores *numpy.array* object with the shape *(n\_samples, q)* where  $q = \begin{pmatrix} d + n \\ d\end{pmatrix}$.


 The second attribute is *Kernel.K* is a interior structure for fast computing of TK. 
