---
layout: post
title: "A quick start guide"
# author: "Aleksandr Talitckii"
categories: facts
tags: [facts,documentation]
image: start.jpg
---

Tessellated Kernel Learning (TKL) is a free machine learning **MATLAB** and **python** toolbox for learning optimal Tessellated Kernel (TK) functions for Support Vector Machine (SVM) classification and regression problems. TKs are a class of kernel functions that are ideal for kernel learning because they admit a linear parameterization (tractability); are dense in the set of all kernels (accuracy); and every member is universal so that the hypothesis space is infinite-dimensional (scalability).

**What can I do with TKL?**

TKL can be used to
1.  Learn optimal TK kernel functions for a given set of inputs x and outputs y.
2.  Train a Support Vector Machine (SVM) using the optimal TK kernel function.
3.  Outputs may be binary (classification) or real valued (regression).

## How to use TKL

If you only want to use TKL-SVM or build a Tessellated kernel, please look at these examples [1](https://talitsky.github.io/v1/example-regression) and [2](https://talitsky.github.io/v1/example-classification), as well as the basic functions for the [kernel](https://talitsky.github.io/v1/svm-training) and the [model construction](https://talitsky.github.io/v1/svm-training). Also you can download our [manual](https://github.com/Talitsky/PMKL/raw/main/TKL_User_Manual.pdf)

## What is Multiple Kernel Learning? 

To whom is interested in what MKL is, a good paper is [Multiple kernel learning algorithms](https://www.jmlr.org/papers/volume12/gonen11a/gonen11a.pdf) *M Gönen* or just read on [wikipedia](https://en.wikipedia.org/wiki/Multiple_kernel_learning).

## How to understand the TKL?

What is TKL and how to use it, you can read one of these articles [1](http://control.asu.edu/Publications/2021/Colbert_NIPS_2021.pdf) or [2](https://arxiv.org/abs/1711.05477). And we also have a shortened version on our [website](https://talitsky.github.io/v1/tkl-intro)

## Technical Support 

Our goal is to make use of TKL as simple as humanly possible. However, our background is not in coding and sometimes we come up short. If you are having a serious technical issue and neither the help commands nor the manual are helping, and believe there is a bug in the program, please report it to: [brendon.colbert@asu.edu](brendon.colbert@asu.edu) or [atalitck@asu.edu](atalitck@asu.edu). If there is a bug, we will add it to the known bug list and do our best to fix it.

Alternatively, if you would like to volunteer for the TKL development team, we would be happy to include you (no compensation - Sorry). Send an email to [mpeet@asu.edu](mpeet@asu.edu)
