---
layout: category
title: Tessellated Kernels
category: documentation
permalink: /documentation
---

## What is Tesselated Kernel?

TK is ...

## How to use it?

Firstly, we realized the KernelFunctions to compute TK matrices and a monomial basis, given $x$ and degree $q$ one  can call

`import numpy as np
from PMKLpy import KernelFunctions
x = np.random.rand(100, 1)
q = 1
Lower, Upper =  x.min(axis = 0) - 0.1 , x.max(axis = 0) + 0.1
Kernel = KernelFunctions.Kernel(x, Lower, Upper,  q)`


