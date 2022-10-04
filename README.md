# tensor-grad

An auto-differentiation engine for arbitrary tensors written from scratch in C++.

The engine can be used to find derivatives of any tensor with respect to any other tensor, and so it can be used as a backpropagation algorithm to train neural networks via gradient descent, which typically requires finding derivatives of a scalar loss with respect to various weight tensors.

An example of usage can be found in `example.cpp`, where `main` has a runtime ~45ms.

TODO:
* Minor refactors and optimizations
* Tensor slicing operation

# Overview

The general idea is that we have some tensor $Y$ of shape $(y_1, \ldots, y_n)$ that has been 'produced' by tensors $(X^{(1)}, \ldots, X^{(k)})$ each of shape $(x_1^{(i)}, \ldots, x_{m_i}^{(i)})$ for $i = 1, \ldots, k$. Each $X^{(i)}$ may also have been produced by some other set of tensors. By 'produced' it is meant that for some function $f$,

$$Y = f(X^{(1)}, \ldots, X^{(k)})$$

We wish to find the derivative of $Y$ with respect to some target tensor $T$ of shape $(t_1, \ldots, t_p)$, i.e. we wish to find the tensor,

$$D_{i_1 \cdots i_n j_1 \cdots j_p}^{(Y)} := \frac{\partial Y_{i_1 \cdots i_n}}{\partial T_{j_1 \cdots j_p}}$$

which has shape $(y_1, \ldots, y_n, t_1, \ldots, t_p)$.

(some of $X^{(i)}$ may depend on this target variable $T$, or there may be no dependence at all, in which case $D_{i_1 \cdots i_n j_1 \cdots j_p}^{(Y)} = 0$ if $Y \neq T$).

In the case of neural networks, this target variable may be a weight/bias matrix, where the tensor we wish to find the derivative of, $Y$, may be a scalar loss of shape $(1)$.

The total derivative of $Y$ is,

$$
\begin{align}
dY_{i_1 \cdots i_n} &= \sum_{i=1}^{k} \sum_{k_1^{(i)} \cdots k_{m_i}^{(i)}} \frac{\partial Y_{i_1 \cdots i_n}}{\partial X_{k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)}} dX_{k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)} \\
\implies \frac{\partial Y_{i_1 \cdots i_n}}{\partial T_{j_1 \cdots j_p}} &= \sum_{i=1}^{k} \sum_{k_1^{(i)} \cdots k_{m_i}^{(i)}} \frac{\partial Y_{i_1 \cdots i_n}}{\partial X_{k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)}} \frac{\partial X_{k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)}}{\partial T_{j_1 \cdots j_p}} \\
D_{i_1 \cdots i_n j_1 \cdots j_p}^{(Y)} &= \sum_{i=1}^{k} \sum_{k_1^{(i)} \cdots k_{m_i}^{(i)}} F_{i_1 \cdots i_n k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)} D_{k_1^{(i)} \cdots k_{m_i}^{(i)} j_1 \cdots j_p}^{(X)}
\end{align}
$$

where $F_{i_1 \cdots i_n k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)} := \frac{\partial Y_{i_1 \cdots i_n}}{\partial X_{k_1^{(i)} \cdots k_{m_i}^{(i)}}^{(i)}}$.

Since $F^{(i)}$ depends only on $X^{(i)}$ and $Y$ for each $i = 1, \ldots, k$ and $Y = f(X^{(1)}, \ldots, X^{(k)})$, the tensor $F^{(i)}$ can be completely determined with knowledge of $f$. In the code, this requires implementing an appropriate function to calculate this tensor for each core operation (addition, unary minus, matrix multiplication, etc.).

Computing $D^{(X)}$ involves repeating the total derivative formula for $X$ as we did for $Y$, and continuing this recursive behaviour until we reach a leaf node. A leaf node represents a tensor that has not been produced by any other tensors, e.g. in machine learning, the weight matrices are typically leaf nodes as they are created through some initialization that does not depend on other tensors (Xavier initialization, etc.). We typically choose these to be the target tensors for which we find derivatives with respect to.

Let $E$ be such a leaf node, then if $E = T$,

$$
\begin{align}
\frac{\partial E_{i_1 \cdots i_p}}{\partial T_{j_1 \cdots j_p}} = \prod_{r = 1}^p \delta_{i_r j_r}
\end{align}
$$

where $\delta_{i j} = 1$ if $i = j$ else $0$.

And otherwise $(E \neq T)$,

$$
\begin{align}
\frac{\partial E_{i_1 \cdots i_n}}{\partial T_{j_1 \cdots j_p}} = 0
\end{align}
$$

# Core Operations

Implementations of the core operations can be found in `tensor/operations`. Since functions can be written as a combination of a handful of core operations, only the derivative information of the core operations need to be implemented.

An example of derivative information for tensor multiplication: 

For two arbitrary tensors $X$, $Y$ of the same shape $(n_1, \ldots, n_k)$, consider $f(X, Y) = X * Y =: Z$ (elementwise multiplication). Then,

$$
\begin{align}
F_{i_1 \cdots i_k j_1 \cdots j_k}^{(1)} := \frac{\partial Z_{i_1 \cdots i_k}}{\partial X_{j_1 \cdots j_k}} &= \frac{\partial (X_{i_1 \cdots i_k} Y_{i_1 \cdots i_k})}{\partial X_{j_1 \cdots j_k}} \\
&= (\prod_{r = 1}^k \delta_{i_r j_r}) Y_{i_1 \cdots i_k}
\end{align}
$$

It then follows that $F_{i_1 \cdots i_k j_1 \cdots j_k}^{(1)} = 0$ for all entries except for when $i_r = j_r$ for all $r = 1, \ldots, k$, where $F_{i_1 \cdots i_k i_1 \cdots i_k}^{(1)} = Y_{i_1 \cdots i_k}$.

And similarly for $F_{i_1 \cdots i_k j_1 \cdots j_k}^{(2)} := \frac{\partial Z_{i_1 \cdots i_k}}{\partial Y_{j_1 \cdots j_k}}$, it follows that $F_{i_1 \cdots i_k j_1 \cdots j_k}^{(2)} = 0$ except for when $i_r = j_r$ for all $r = 1, \ldots, k$, where $F_{i_1 \cdots i_k i_1 \cdots i_k}^{(2)} = X_{i_1 \cdots i_k}$.