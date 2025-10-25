# Components of a convolutional network

![[Pasted image 20251022092118.png]]

## Batch normalization

We want to normalize the outputs of a layer so they have zero mean and unit variance.

Why? It helps reducing "internal covariate shift", so it improves optimization.

We can normalize a batch of activations like this:
$$ \hat x=\frac{x-E[x]}{\sqrt{Var[x]}} $$
This is a **differentiable function**, so we can use it as on operator in our networks and backprop through it.

We also compute the empirical mean and variance independently for each dimension:
- $\mu_j=\frac1N\sum\limits_{i=1}^N x_{i,j}$
- $\sigma_j^2=\frac1N\sum\limits_{i=1}^N(x_{i,j}-\mu_j)^2$
- $\hat x_{i,j}=\Large\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}$

>[!note] Personal note
>I don't know if these formulas mean anything but here's the screenshot.
>![[Pasted image 20251022104713.png]]

![[Pasted image 20251022115537.png]]

Batch normalization:
- Improves gradient flow through the network
- Allows higher learning rates
- Reduces the strong dependence on initialization
- Acts as a form of regularization

![[Pasted image 20251022120144.png]]

It also:
- makes deep networks much easier to train;
- allows higher learning rates, faster convergence;
- makes networks become more robust to initialization;
- acts as regularization during training;
- has zero overhead at test-time (can be fused with conv).

>[!warning]
It also behaves differently during training and testing: this is a very common source of bugs.


## Activation functions

![[Pasted image 20251022131148.png]]

These are the most common variation functions:
![[Pasted image 20251022131800.png]]

### Sigmoid functions

![[Pasted image 20251022131921.png]]
$$ \sigma(x)=\frac1{1+e^{-x}} $$

They squash numbers to range [0,1].
Sigmoid functions are since they have nice interpretation as a saturating “firing rate” of a neuron.

3 problems:
1. Saturated neurons “kill” the gradients.
2. Sigmoid outputs are not zero-centered.
3. exp() is a bit compute expensive.

### tanh(x)

![[Pasted image 20251022132734.png]]

>[!note] Personal note
>This is basically a sigmoid shifted down.

It squashes numbers to range [-1,1].

**Pro**: it's zero-centered.
**Con**: still kills gradients when saturated.

### ReLU

![[Pasted image 20251022132942.png]]

Computes:
$$ f(x)=\max(0,x) $$

**Pros**:
- Does not saturate (in +region)
- Very computationally efficient
- Converges much faster than sigmoid/tanh in practice (e.g. 6x)
- Actually more biologically plausible than sigmoid.

**Cons**:
- Not zero-centered output
- An annoyance: what is the gradient when x < 0?

>[!info]
>People like to initialize ReLU neurons with slightly positive biases (e.g. 0.01).

### Leaky ReLU

![[Pasted image 20251022133542.png]]

$$ f(x)=\max(0.01x, x) $$

**Pros**:
- Does not saturate
- Computationally efficient
- Converges much faster than sigmoid/tanh in practice! (e.g. 6x)
- will not “die”.

It can be generalized to **Parametric Rectifier (PReLU)**:
$$f(x)=\max(\alpha x,x)$$

### Exponential Linear Units (ELU)

![[Pasted image 20251022133810.png]]

$$f(x)= \begin{cases}
x \qquad\qquad\qquad\quad if\ \ x>0\\ 
\alpha(e^x- 1)\quad\ \ if\ \ x\le0 
\end{cases}$$
**Pros**:
- All benefits of ReLU
- Closer to zero mean outputs
- Negative saturation regime compared with Leaky ReLU adds some robustness to noise.

**Con**: Computation is also exponential.

## Rule of thumb: in practice

- Don’t think too hard. Just use ReLU.
- Try out Leaky ReLU / ELU (SELU) if you need to squeeze that last 0.1%
- Don’t use sigmoid or tanh

# Data preprocessing

## Step 1: Preprocess the data

![[Pasted image 20251022172428.png]]

Consider what happens when the input to a neuron (x) is always positive:
$$ f\bigg(\sum\limits_i w_ix_i+b\bigg) $$
The gradients on w are always either all positives or all negatives.
This doesn't happen when we have zero-mean data and that's why this property is so important.

Visual example:
![[Pasted image 20251022173113.png | 400]]

In practice, we may also see **PCA** and **Whitening** of the data:
![[Pasted image 20251022173344.png]]

**Before normalization**: classification loss is very sensitive to changes in weight matrix; thus, it's hard to optimize.
![[Pasted image 20251022173512.png]]

**After normalization**: it's less sensitive to small changes in weights; thus, it's easier to optimize.
![[Pasted image 20251022173625.png]]

>[!info]
>**Rule of thumb for images**: center only.

# Weight initialization

What happens when W=0 initialization is used (and b=0)?
![[Pasted image 20251022173910.png]]

Then, all outputs are 0 and all gradients are the same. This means there's no actual learning.

An idea is to start off with small random numbers (more specifically, a gaussian with zero-mean and 10^-2 standard deviation).

Example:	
	w = 0.01 * np.random.randn(Din, Dout)

>[!info] Outcome
>It works for small networks, but has problems with deeper networks.

## Weight initialization: activation statistics

>[!note] Personal note
>Look from page 45 to page 52 in the slides.

## Weight initialization: Xavier initialization

Xavier initialization corresponds to: std=1/sqrt(Din)

If the weights are:
- i.i.d.;
- independent (very important);
- zero-mean
then, **variance input = variance output**.

In other words:
$$ Var(w_i)=\frac1{Din} \implies Var(y_i)=Var(x_i) $$

# Regularization

It is a technique used to improve single model performance.

## Add term to loss
The idea is to **add a term to the loss function**:
$$ L=\frac1N\sum\limits_{i=1}^N\sum\limits_{j\not =y_i}\max(0, f(x_i; W)_j-f(x_i;W)_{y_i}+1)+\color{red} \lambda R(W) $$

In common use:
- **L2 regularization**: $R(W)=\sum\limits_k\sum\limits_l W^2_{k,l}$ (weight decay).
- **L1 regularization**: $R(W)=\sum\limits_k\sum\limits_l |W_{k,l}|$.
- **Elastic net (L1+L2)**: $R(W)=\sum\limits_k\sum\limits_l \beta W^2_{k,l} +|W_{k,l}|$.

## Dropout

In each forward pass, randomly set some neurons to zero.

The probability of dropping is a hyperparameter, it's usually set to 0.5.

![[Pasted image 20251022201552.png]]

This technique works it allows to perform better classifications in case of exceptions (so a certain class can be identified even if not all the criteria are met).

Example:
![[Pasted image 20251022201859.png]]

Although, dropout makes our output random, so we want to "average out" some randomness at test-time:
$$ y=f(x)=E_z[f(x,z)]=\int p(z)f(x,z)dz $$
However, this integral is complex to compute.

## Test time

We want to find a way to approximate the integral shown before.

So, let's consider a single neuron:
![[Pasted image 20251025091729.png]]

At test time we have:
$$ E[a]=w_1x+w_2x $$
During training we have:
$$ \begin{gather}
E[a]=\frac14(w_1x+w_2y)+\frac14(w_1x+0y)+\frac14(0x+0y)+ \frac14(0x+w_2y)= \\ =\frac12(w_1x+w_2y) 
\end{gather} $$
>[!info]
>So, during the training phase we multiplied the expectation function by dropout probability (which is 0.5 in this case) without knowing.

At test time all neuron are always active, so we must scale the activation so that for each neuron: **output at test time = expected output at training time**.

## A common pattern

Usually regularization works like this:
- **Training**: add some kind of randomness $$ y=fw(x,z) $$
- **Testing**: average out randomness (sometimes approximate) $$ y=f(x)= E_z[f(x,z)]=\int p(z)f(x,z)dz $$
For example, with **batch normalization**:
- **Training**: normalize using stats from random minibatches.
- **Testing**: use fixed stats to normalize.

>[!info]
>For ResNet and later, often L2 and Batch Normalization are the only regularizers.

## Data augmentation

>[!note] Personal note
>I don't know what to get from this part. See from page 78 to page 92 on the slides.

# Training neural networks

## Overview

1. **One time setup**
   activation functions, preprocessing, weight initialization, regularization.
2. **Training dynamics**
   babysitting the learning process, parameter updates, hyperparameter optimization.
3. **After training**
   model ensembles, transfer learning.

## Choosing a learning rate

![[Pasted image 20251025102305.png]]

## Learning rate decay

>[!note] Personal note
>Check the slides from page 98 to page 103.

## How long to train? Early stopping

![[Pasted image 20251025103215.png]]

# Hyperparameter optimization

>[!note] Personal note
>Check the slides from page 105 to the end. I got a bit fed up about this lecture.

