# Gradient descent

Gradient descent is a **first-order** iterative minimization algorithm. The general idea is to move where the function decreases the most.

Let's take an example of loss function: $\ell_\Theta : \Bbb R^2 \to \Bbb R$

The steps are:
1. Start from some point $\Theta^{(0)}\in \Bbb R^2$.
2. Iteratively compute: $$\Theta^{(t+1)}=\Theta^{(t)}-\alpha \nabla \ell_{\Theta^{(t)}}$$
3. Stop when the minimum is reached.


For the whole function the equation is: $$x^{(t+1)}=x^{(t)}-\alpha \nabla f(x^{(t)})$$

The gradient requires $f$ to be differentiable at all points.
>[!info] Recall
>- $f$ has partial (or even directional) derivatives $\cancel\implies$ $f$ is differentiable
>- $f$ has **continuous gradient** $\implies$ $f$ is differentiable

## Stationary points

A **stationary point** is a point such that: $$x^{(t+1)}=x^{(t)}$$
So, a point where: $$\alpha\nabla f(x^{(t)})=0$$

Gradient descent tends to "get stuck" at stationary points. However, a stationary point isn't necessarily a local minimum.

## Learning rate

The parameter $\alpha$ (which is always $a>0$) in ML context is also called **learning rate**. 

So, the length of a step for the gradient descent is $\alpha\| \nabla f\|$.

When taking a step various things can happen:
- **Too small**:  slow convergence speed.
- **Too big**: risk of overshooting.
- Optimal values can be found via line search algorithms.

![[Pasted image 20251016122823.png]]

So, in the end, all we have to do is to find the right $\alpha$ value so that: 
$$\arg \min_\alpha f (x^{(t)}-\alpha\nabla f(x^{(t)}))$$

## Decay and momentum

The learning rate can be **adaptive** or follow a **schedule**.

Decrease $\alpha$ according to a **decay** parameter $\rho$.

Examples:
$$\alpha^{(t+1)}=\frac{\alpha^{(t)}}{1+\rho t}, \qquad \qquad \alpha^{(t+1)}=\alpha^{(0)}e^{-\rho t}$$

Accumulate past gradients and keep moving in their direction:
$$v^{(t+1)}=\lambda v^{(t)}-\alpha\nabla f(x^{(t)})$$
$$x^{(t+1)}=x^{(t)}+v^{(t+1)}$$
>[!info]
>During this process we added two parameters:
>- $v^{(t)}$, which is the **velocity**
>- $v^{(t+1)}$, which is the **momentum**

The $\lambda$ parameter allows to accumulate the previous update, so we can accelerate the research of a local minimum.

So, the step length is proportional to: $$\frac1{1-\lambda}\alpha \|\nabla f\|$$
The bigger is $\lambda$ the higher is the acceleration and, thus, the more likely is to escape a "meaningless" local minimum.

## First Order Acceleration Methods

Let us try to unroll gradient descent:
$$
\begin{gather}
x^{(t+1)}=x^{(t)}-\alpha\nabla f(x^{(t)}) \\ \ \\ \ \\
x^{(1)}=x^{(0)}-\alpha\nabla f(x^{(0)}) \\
x^{(2)}=x^{(1)}-\alpha\nabla f(x^{(1)}) \\
\qquad \qquad \qquad \quad = x^{(0)}-\alpha\nabla f(x^{(0)})-\alpha\nabla f(x^{(1)})
\\ . \\ . \\ . \\
x^{(t+1)}=x^{(0)}-\alpha\sum\limits_{i=1}^t \nabla f(x^{(i)})
\end{gather}
$$
And, if we consider momentum:
$$ x^{(t+1)}=x^{(0)}+\alpha\sum\limits_{i=1}^t \frac{1-\lambda^{t+1-i}}{1-\lambda} \nabla f(x^{(i)}) $$

We can also use the more general form:
$$ x^{(t+1)}=x^{(0)}+\alpha\sum\limits_{i=1}^t \gamma_i^t \nabla f(x^{(i)}) \qquad for\ some\ \gamma_i $$
And for diagonal matrices:
$$ x^{(t+1)}=x^{(0)}+\alpha\sum\limits_{i=1}^t \Gamma_i^t \nabla f(x^{(i)}) \qquad for\ some\ diag.\ matrix\ \Gamma_i $$

## Gradient descent for deep learning

Gradient descent can be applied to **non-convex** problems too, but without optimality guarantees.

So, in order to gain generalization, we don't care about the **global** optimum, but we mainly focus on a **local** optimum that's good enough.

>[!info]
>Generally speaking, we're interested in solutions that are **efficient** and **numerically stable**.

In the general DL setting, each parameter gets updated so as to **decrease loss**:
$$ \theta_i \leftarrow \theta_i-\alpha\frac{\partial\ell}{\partial\theta_i} $$
>[!note]
>The left arrow represents the update operation.

The gradient tells us how to modify the parameters.
- $\theta$ stores the neural network parameters, possibly **millions**.
- The loss can be **non-convex** and **non-differentiable** (cannot even apply gradient descent!).
- Must address **computational complexity** and **numerical stability**.

## Stochastic gradient descent

Recall that the loss is usually defined over $n$ training examples:
$$
\begin{gather}
\ell_\Theta(\{x_i, y_i\})=\frac1n\sum\limits_{i=1}^n (y_i-f_\Theta(x_i))^2 \\ \ \\
\ell_\Theta(\{x_i, y_i\})=\frac1n\sum\limits_{i=1}^n \hat \ell_\Theta (\{x_i, y_i\})
\end{gather}
$$
which requires computing the gradient for each term in the simulation:
$$ \nabla\ell_\Theta(\{x_i, y_i\})=\frac1n\sum\limits_{i=1}^n  \nabla \hat\ell_\Theta (\{x_i, y_i\}) $$

Two **bottlenecks** make gradient descent impractical:
- Number of examples.
- Number of parameters.

We can also rewrite the loss gradient like this:
$$  \nabla\ell_\Theta(\mathcal T)=\frac1n\sum\limits_{i=1}^n  \nabla \hat\ell_\Theta (\mathcal T)  $$
Anyway, computing the gradient loss for all the data may take too much time and resources; so, we can compute it for a small representative subset of $m\ll n$ examples:
$$ \frac1m\sum\limits_{i=1}^m  \nabla \hat\ell_\Theta (\mathcal B) \approx \frac1n\sum\limits_{i=1}^n  \nabla \hat\ell_\Theta (\mathcal T) $$

The **mini-batch** $\mathcal B \subset \mathcal T$ is drawn uniformly.

The true gradient $\nabla\ell_\Theta$ is approximated, but with a significant **speed-up**.

The **Stochastic gradient descent** algorithm is as follows:
1. Initialize parameters $\theta$.
2. Pick a mini-batch $\mathcal B$.
3. Update with the downhill step (use momentum if desired):
   $$ \theta\leftarrow\theta-\alpha\nabla\ell_\Theta(\mathcal B) $$
4. Go back to step 2.

When steps 2-4 cover the entire training set $\mathcal T$ we have an **epoch**.
>[!info]
>Epoch just means that all the data were used to update the model at least once.

Like gradient descent, the algorithm proceeds for many epochs.

>[!warning]
>The update cost is constant regardless of the size of $\mathcal T$.

### Comparison with normal gradient descent

Unlike GD, SGD does not depend on the number of examples, implying better generalization.

### Some practical considerations

Stochastic gradient descent:
- Needs the training data to be **shuffled** to avoid data bias (or otherwise the approximation won't work).
- Each mini-batch can be processed in parallel.
- If the dataset is large, more steps will be needed to reach convergence.
	- It has a rapid initial progress but a slow asymptotic convergence.
- Small mini-batches can offer a regularizing effect.
	- Very small batches → high variance in the estimation of the gradient → use a small learning rate to maintain stability.

SGD can find a **low value** of the loss quickly enough to be useful, even if it's not a minimum.

# The perceptron

It's the elementary unit of a neural network and it's inspired by the human neuron.

For each perceptron:
- Inputs are **feature values** (which is $x$).
- Each feature has a **weight** (which is $w$).
- Sum is the **activation**.

Thus, the formula is: $$ activation_w(x)=\sum\limits_iw_i\cdot x=w\cdot x $$
## Linear classifier

If the perceptron is a linear classifier, then we'll have:
- output +1 if the activation is positive;
- output -1 if the activation is negative.

The goal of linear classification is to distinguish correctly between two classes (or more if we generalize), like this example:
![[Pasted image 20251018150909.png]]

So, the process is as follows:
- Start with weight = 0.
- For each training instance:
	- Classify with current weight	  $$ y=\begin{cases} +1,\qquad if\quad w\cdot x\ge0 \\  -1,\qquad if\quad w\cdot x\lt0\end{cases}$$
	- If correct (i.e., $y=y^*$), no change is needed.
	- If wrong: adjust the weight vector by adding or subtracting the feature vector, in this way: $$ w=w\pm y^*\cdot x $$
		- It's + if $y^*=1$
		- It's - if $y^*=-1$.
Example:
![[Pasted image 20251018153112.png]]

## Perceptron convergence

The perceptron algorithm is pretty powerful: if the dataset is linearly seperable, the perceptron will find a separating hyperplane in a **finite number of updates**.
However, if the dataset is not linearly seperable, then the perceptron will **loop forever**.

>[!note] Personal note
>I don't know how to go further without getting into infinite and useless details, so I'll just put this screenshot: ![[Pasted image 20251018154404.png]]

## Improving the perceptron

The idea is to abandon the deterministic approach (left) and rather adopt a probabilistic one (right).

![[Pasted image 20251018154648.png]]

![[Pasted image 20251018155035.png]]

## Logistic regression

The perceptron scoring is: $z= w\cdot x$.
If $z= w\cdot x$ is high, there's a high chance the output will be positive (output = 1). If $z= w\cdot x$ is low, there's a high chance the output will be negative (output = 0).

For logistic regression we use the **Sigmoid function**:
$$\phi(z)=\frac1{1+e^{-z}}$$
![[Pasted image 20251018171443.png]]

>[!info]
>The curve represents the probability of getting either a 0 or a 1.

### Multiclass logistic regression

Recall perceptron:
- A weight vector for each class: $w_y$.
- Score (activation) of a class y: $w_y\cdot x$.
- Prediction highest score wins: $y=\arg \max_y w_y\cdot x$.

And the activation functions become:
$$
z_1, z_2, z_3 \to \frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}}, \frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}}, \frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}}
$$

### How to find the best w?

We do it by computing **Maximum likelihood estimation**:
$$
\max_w\ ll(w)=\max_w\sum\limits_i\log P(y^{(i)}|x^{(i)};w)
$$

But we have to take into account that $P$ is different if the regression is multi-class or not.

Simple regression:
- $P(y^{(i)}=+1|x^{(i)};w)=\Large\frac1{1+e^{-w\cdot f(x^{(i)})}}$.
- $P(y^{(i)}=-1|x^{(i)};w)=1-\Large\frac1{1+e^{-w\cdot f(x^{(i)})}}$
Multi-class regression:
- $P(y^{(i)}|x^{(i)};w)=\Large\frac{e^{w_{y^{(i)}}\cdot f(x^{(i)})}}{\sum\limits_i e^{w_y\cdot f(x^{(i)})}}$
## SGD and perceptron

>[!note] personal note
>I got a headache trying to read this:
>![[Pasted image 20251018174121.png]]

# Neural network

![[Pasted image 20251018175316.png]]

## The need of priors

Deep feed-forward networks are probably universal.

However:
- We can make them arbitrarily complex.
- The number of parameters can be huge.
- Very difficult to optimize.
- Very difficult to achieve generalization.

## Advanced models

In practice, many interesting phenomena are highly nonlinear. So, we must choose a good learning model to capture these phenomena.

A powerful model should be as universal as possible.

### Deep composition

The simplest example of a nonlinear parametric model: $$ f\circ f(x) $$
>[!info]
>$f(x)$ on its own is linear but $f\circ f(x)$ is not.

If instead of a linear function we use a logistic function we obtain: $$ \sigma\circ f(x) $$
And, thus, we have a logistic regression model.

And it works like this:
![[Pasted image 20251018181428.png]]

Actually we can use other activation functions than logistic:
![[Pasted image 20251018181538.png]]

### Multi-layer perceptron

We call the composition with linear $f$ and nonlinear $\sigma$: $$ g_\Theta(x)=(\sigma\circ f_{\Theta_{n}}) \circ (\sigma\circ f_{\Theta_{n-1}}) \circ \ ... \ \circ (\sigma\circ f_{\Theta_{1}})(x) $$
a **multi-layer perceptron** (MLP) or **deep feed-forward neural network**.

The parameters of weights of the MLP are scattered across the layers.

Each layer outputs an intermediate **hidden representation**: $$ x_{\ell+1} = \sigma_\ell(W_\ell x_\ell+b_\ell) $$
where we encode the weights at layer $\ell$ in the matrix $W_\ell$ and bias $b_\ell$.

The bias can be integrated inside the weight matrix by writing:
$$ W\mapsto\begin{pmatrix}W & b\end{pmatrix}, \quad x\mapsto \begin{pmatrix}x \\ 1\end{pmatrix}$$
because each $f$ is linear in the parameters just like in linear regression.

#### Hidden units

At each hidden layer we have:
$$ x_{\ell+1} = \sigma_\ell(W_\ell x_\ell) $$

Each row of the weight matrix is called a **neuron** or **hidden unit**:
$$ W_x= \begin{pmatrix}\textemdash unit\textemdash \\ \vdots \\ \textemdash unit\textemdash \end{pmatrix}\begin{pmatrix} \textbar \\ x\\ \textbar\end{pmatrix} $$

We have two interpretations:
1. Each layer is a vector-to-vector function $\Bbb R^p\to \Bbb R^q$.
2. Each layer has $q$ units acting in **parallel**.
   Each unit acts as a scalar function $\Bbb R^p\to \Bbb R$.

#### Single layer illustration
![[Pasted image 20251018212842.png]]

#### Output layer

The output layer determines the co-domain of the network:
$$y= (\sigma\circ f)\circ (\sigma\circ f)\circ \dots\circ (\sigma\circ f)(x)$$

If $\sigma$ is the logistic sigmoid, then the entire network will map:
$$ \Bbb R^p\to (0,1)^q $$

For generality, it is common to have a linear layer at the output:
$$ y= f\circ (\sigma\circ f)\circ \dots\circ (\sigma\circ f)(x) $$
mapping:
$$\Bbb R^p\to \Bbb R^q$$

### Deep ReLU networks

Adding a linear layer at the output:
$$y= f\circ \sigma(\dots)(x)$$
expresses $y$ as a combination of "ridge functions" $\sigma(\dots)$.

The ReLU activation function is $\sigma(x) = \max\{0,x\}$ and it looks like this:
![[Pasted image 20251018214936.png]]

## Universality

What class of functions can we represent with a MLP?

If $\sigma$ is sigmoidal, we have the following:
>[!info] Theorem
>The Universal Approximation Theorem states that a **feedforward neural network with a single hidden layer**, containing a **finite number of neurons**, and using a **non-linear, bounded, and continuous activation function** (like sigmoid or tanh), can **approximate any continuous function** on a compact subset of **Rⁿ** to any desired degree of accuracy.

The network in the theorem has just one hidden layer.

For large enough $q$, the training error can be made arbitrarily small.

UAT theorems exist for other activations like ReLUs and locally bounded non-polynomials.

These proofs are **not constructive**.
They do not say how to compute the weights to reach a desired accuracy.

Some theorems give bounds for the width q ("# of neurons").

Some theorems show universality for > 1 layers (deep networks).

In general, we deal with nonconvex functions. Empirical results show that large $q$ + gradient descent leads to very good approximations.

## Training

Given a certain MLP, the activity of solving its MSE loss for the weights $\Theta$ is known as **training**.

MSE loss to solve:
$$ \ell_\Theta(\{x_i,y_i\})=\frac1n\sum\limits^n_{i=1} \|y_i- g_\Theta(x_i)\|_2^2 $$

In general, the loss is **not convex** w.r.t. $\Theta$.

As we have seen (I don't remember where), the following special cases are convex:
- One layer, no activation, MSE loss (i.e., linear regression).
- One layer, sigmoid activation, logistic loss (i.e., logistic regression).

However, computing the gradient-descent is highly inefficient and we want to find a way to automatize this process in order to make it more efficient.

## Computational graphs

Consider a generic function $f: \Bbb R \to \Bbb R$.

A **computational graph** is a directed acyclic graph representing the computation of $f(x)$ with **intermediate** variables.

Examples:
![[Pasted image 20251020202830.png]]
![[Pasted image 20251020202915.png]]

## Automatic differentiation

>[!note] Personal note
>See pages 104-113 for this one.

## Back-propagation

>[!note] Personal note
>See from 114 to the end.



