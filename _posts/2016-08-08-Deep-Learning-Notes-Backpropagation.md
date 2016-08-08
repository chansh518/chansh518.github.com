---
layout: post
title: "Deep Learning深度学习（二）：反向传播"
description: "神经网络的训练目标是最小化目标函数$$C$$，使用梯度下降的方法来训练参数$$w$$和$$b$$。在梯度下降中，需要根据$$\frac{\partial C}{\partial w}$$和$$\frac{\partial C}{\partial b}$$的计算结果来决定参数$$w$$和$$b$$的改变幅度。反向传播便提供了一种计算偏导的方法。"
category: "Deep Learning"
tags: Deep Learning, Backpropagation
---

> 本文为[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)一书的学习笔记，仅记录书中的理论部分，省略了实验和例子，用到的图都来自于书中。
>
> 这本书很适合初学者读，作者写得非常详细，简单易懂，里面涉及到的内容都有解释，还有配有代码和例子，强烈推荐！

神经网络的训练目标是最小化目标函数$$C$$，使用梯度下降的方法来训练参数$$w$$和$$b$$。在梯度下降中，需要根据$$\frac{\partial C}{\partial w}$$和$$\frac{\partial C}{\partial b}$$的计算结果来决定参数$$w$$和$$b$$的改变幅度。反向传播便提供了一种计算偏导的方法。

在前一章里已经讲过，每个神经元的计算过程可以写作：

$$
\begin{align}
z^l&=w^l a^{l-1}+b^l, \\
a^l&=\sigma(z^l).
\end{align}
$$

$$a^l$$为第l层的神经元的输出的向量，$$w$$和$$b$$为参数，$$\sigma$$为非线性的激活函数，如sigmoid。

给整个神经网络设定一个目标函数为$$C$$（Cost Function），例如：

$$
C=\frac{1}{2n}\sum_x\parallel y(x)-a^L(x)\parallel^2
$$

$$y(x)$$是样本的label，$$a^L(x)$$是输出层输出的结果。

在用梯度下降法求使目标函数$$C$$最小的参数时，每一步参数的更新方法是：

$$
w_k \to w_{k}' = w_k-\frac{\eta}{m}\sum_j \frac{\partial C_{x_j}}{\partial w_k}, \\
b_k \to b_k' = b_k-\frac{\eta}{m}\sum_j \frac{\partial C_{x_j}}{\partial b_k}.
$$

其中$$\eta$$是步长，每个mini-batch包含m个训练数据。下面通过反向传播来计算$$\frac{\partial C}{\partial w}$$和$$\frac{\partial C}{\partial b}$$。

为方便计算，定义残差(error)：

$$
\delta_j^l\equiv\frac{\partial C}{\partial z_j^l}
$$

$$z_j^l$$表示l层第j个神经元的input的加权和。根据定义，可以推算出每一层残差的计算方法：

$$
\begin{eqnarray}
\delta_j^L=\frac{\partial C}{\partial a_j^L}\frac{\partial a_j^L}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\sigma'(z_j^L) \tag 1
\end{eqnarray}
$$

$$
\begin{eqnarray}
\delta_j^l&=&\sum_k\frac{\partial C}{\partial z_k^{l+1}}\frac{\partial z_k^{l+1}}{\partial z_j^l}\\
&=&\sum_k\frac{\partial z_k^{l+1}}{\partial z_j^l}\delta_k^{l+1}\\
&=&\sum_k\frac{\partial (\sum_i w_{ki}^{l+1}a_i^l+b_k^{l+1})}{\partial z_j^l}\delta_k^{l+1}\\
&=&\sum_kw_{kj}^{l+1}\frac{\partial a_j^l}{\partial z_j^l}\delta_k^{l+1}\\
&=&\sum_kw_{kj}^{l+1}\delta_k^{l+1}\sigma'(z_j^l) \tag 2
\end{eqnarray}
$$

在此基础上，得到我们关心的$$\frac{\partial C}{\partial w_{jk}^l}$$和$$\frac{\partial C}{\partial b_{j}^l}$$的计算方法，用error表示为：

$$
\begin{eqnarray}
\frac{\partial C}{\partial w_{jk}^l}&=&\frac{\partial C}{\partial z_{j}^{l}}\frac{\partial z_j^l}{\partial w_{jk}^l}\\
&=&\delta_j^l a_{k}^{l-1}
\end{eqnarray} \tag 3
$$

$$
\begin{eqnarray}
\frac{\partial C}{\partial b_{j}^l}&=&\frac{\partial C}{\partial z_{j}^{l}}\frac{\partial z_j^l}{\partial b_{j}^l}\\
&=&\delta_j^l
\end{eqnarray} \tag 4
$$

根据上面4个公式，可以从output层往input层计算出每一层的参数$$w$$和$$b$$在进行梯度下降时应该下降的幅度。

### 为什么说backpropagation很快？
试想如果不使用backpropagation方法，一个直观的做法是：

$$
\frac{\partial C}{\partial w_j} \approx \frac{C(w + \epsilon e_j) - C(w)}{\epsilon}
$$

其中$$\epsilon$$是一个极小正数，$$e_j$$是$$j$$方向上的单位向量。用这个方法计算每个参数每一次迭代时调整的方向和大小，这意味着假设整个网络共$$L$$层，每层$$n$$个全连接的节点，那么一共有$$(L-1)n^2$$个$$w$$变量和$$(L-1)n$$个$$b$$变量，梯度下降法中，每一次迭代（假设mini-batch size为$$m$$，即根据$$m$$个样本的情况对所有变量更新一次）就要遍历整个网络$$(L-1)(n+1)nm$$次。相比之下，backpropagation中一次迭代只需要对整个网络进行2次遍历（一次feedforwarding，一次backpropagation），节省了非常多的时间。

> Written with [StackEdit](https://stackedit.io/).