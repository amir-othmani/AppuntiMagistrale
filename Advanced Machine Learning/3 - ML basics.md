# Notation

- $x$ is the input space, and $y$ is the output space.
- In classification problems, the output space $y$ is comprised of $K$ possible classes (or categories).
- For simplicity, we'll just call them $y := {1, 2, ..., K}$
	- when K = 2 we typically use y = {0, 1} or y = {-1, +1}.
- Labeled example: $(x,\ y) \in x\times y$. Interpretation:
	- $x$ represents the description or measurements of an object.
	- $y$ is the category to which that object belongs.

# Loss

Assume there's a distribution $D$ of the space of the labeled examples $X\times Y$.
$D$ is unknown, but it represents the population we care about.

Assume also that the labels are produced by a correct labeling function $f:\ X \rightarrow Y$ and that $Y=f(X)$.

We define the **loss** of the classifier h as the function that measures the error between the assigned prediction and the prediction of the correct labeling function:  $$L(h(X), f(X)) = L(h(X), Y) = \begin{cases} 1, \quad if\enspace h(X) \not = Y \\ \ \\ 0, \quad if\enspace h(X)  = Y \end{cases}$$
>[!info]
>You have to imagine that h(X) is our model and f(X) is a sort of ground truth.

# True risk

The **true risk** of a classifier $h$ is the probability that it does not predict the correct label on a random data point drawn from $D$. That is: $$R_D(h):= \Bbb P[h(x)\not =f(x)]=\Bbb P[h(x)\not =Y]$$
In other words, the **true risk** of the learner $h$ is the expected value of the loss function. $$R_D(h):=\Bbb E_{X\backsim D}\{L(h(X),Y)\} $$
>[!info]
>In practice, we won't have access to the true risk value, but we'll try to optimize the empirical risk instead.

# Bayes classifier

The Bayes classifier works like this:
- our goal is to minimize the true risk of our classifier $h$;
- according to Naive Bayes theorem, this goal can be achieved by maximizing the correct predictions; in other words find the $h$ classifier so that: $$h(x) = \arg \max_{y\in Y} \Bbb P[Y=y|X=x]$$
>[!info]
>$\Bbb P$ stands for "Prediction". It indicates how many predictions are found with the specified argument.

## Bayes risk

Let $h^*$ be the Bayes classifier.
The risk of the Bayes classifier (a.k.a. **Bayes risk**), is the lowest risk you can achieve with a classifier, so: $$R_D(h^*)\le R_D(h)\quad \forall h\in H$$
In other words, **you cannot improve over the Bayes risk**, no matter what ML algorithm you use.

# Batch learning

In practical cases we don't know f (the ground truth) and we need to estimate $h$ from the data.

**Learner input**:
- Training data, $S={(x_1, y_1)...(x_m, y_m)}\in (X\times Y)^m$.

**Learner output**:
- Prediction rule, $h: X\rightarrow Y$.

So, what should be the goal of the learner (classifier)?
Intuitively, $h$ should be correct on future examples.

## Batch learning - IID condition

When is there any hope to find a classifier with high accuracy?

To address this issue, we can make this assumption: all the data $(x_1, y_1)...(x_n, y_n)$ are **identically and independently distributed (i.i.d.)** random labels with distribution $D$, i.e., an i.i.d. sample from $D$.
![[Pasted image 20251009093238.png]]

This assumption is the connection between what we've seen in the past to what we expect to see in the future.
>[!personal note]
>How? I don't know.

# Empirical risk

Training data $S={(x_1, y_1)...(x_m, y_m)}\in (X\times Y)^m$ is an i.i.d. sample from some fixed but unknown probability distribution $D$ over space of labeled examples $X\times Y$.

A learning algorithm takes $S$ as input and returns a predictor $h: X\rightarrow Y$.

Now, we can calculate the **empirical risk** $R_S$ over this sample of data. $$R_S(h)=\frac1m\sum\limits^m_{i=1}\Bbb 1[h(x_i)\not =y_i]$$
In other words, it's the sum of the loss function for each piece of data. In fact the formula can be also written in this way: $$R_S(h)=\frac1m\sum^m_{i=1}L(h(x_i),y_i)$$
>[!info]
>Basically, the key difference between true and empirical risk is the fact that $R_D$ takes as input random elements from all the existent data, while $R_S$ takes as input random elements from only the training data (which is a small sample of all the existent data).

**Question**: In this setting, can any learning algorithm always provide a non-trivial guarantee on the error of the predictor it returns?

**Answer**: No, some assumptions/conditions are required.

# Can only be probably correct

The risk cannot be nullified, it can be arbitrary small, but it never becomes 0.
Even when $R_D(h^*)=0$ we can never obtain $R_D(h)=0$.

So, we're fine with $R_D(h)\le\varepsilon$, where $\varepsilon$ is user-specified.

# Can only be approximately correct

Recall that the input to the learner is randomly generated.
There's always a (very small) chance to see the same example again and again.

So, the problem is: no algorithm can guarantee $R_D(h)\le\varepsilon$.
Then, we can adopt this solution: we allow the algorithm to fail with probability $\delta$, where $\delta\in(0,1)$ is user-specified.

# Probably Approximately Correct (PAC) Learning

It works in this way:
- The learner doesn’t know $\mathcal{D}$ and the Bayes predictor.
- The learner receives accuracy parameter $\epsilon$ and confidence parameter $\delta$.
- The learner can ask for training data $S$ containing $m(\epsilon, \delta)$ examples (that is, the number of examples can depend on the value of $\epsilon$ and $\delta$ but it can’t depend on $\mathcal{D}$ or the Bayes function).
- Learner should output a hypothesis $h$ such that $\mathbb{P}[R_{\mathcal{D}}(h) \leq \epsilon] \geq 1 - \delta$.
- That is, the learner should be **Probably** (with probability at least $1 - \delta$) **Approximately** (up to accuracy $\epsilon$) **Correct**: thus, **PAC Learning**.

## PAC Learning

>[!personal note]
>In the slide there is a formal definition, which I didn't quite understand. What I understood while listening to the recording is: the model needs way more data than the parameters it has (at least 10 times more).

# Things can go wrong

Let's take an example of a regression by using a polynomial curve.
The curve has this (general) formula: $$y(x, \textbf w)=w_0+w_1x+w_2x^2+ ... +w_Mx^M = \sum\limits_{j=0}^Mw_jx^j$$
And the data we have is this:
![[Pasted image 20251010101453.png | 650]]
The green curve is the correct regression (we can use it as a sort of ground truth).

So, our goal is to choose the correct maximum degree (M) for our polynomial.
Let's what happens in different cases:
- **M=0**:
  the curve will look like this
  ![[Pasted image 20251010101913.png | 500]]
  which is obviously wrong (the curve is **underfit**).
- **M=1**:
  ![[Pasted image 20251010102052.png | 500]]
  the curve is still underfit.
- **M=3**:
  ![[Pasted image 20251010102200.png | 500]]
  the curve is correctly fit (it's super close to the actual regression).
- **M=9**:
  ![[Pasted image 20251010102254.png | 500]]
  now the curve is **overfit**. This happened because we gave too many degrees of freedom and, therefore, the curve perfectly fits the training data; but this is terrible because now the model will make bad predictions for future data.

So, when we try to build our model, our goal is to find the right balance between:
- accuracy to the training data,
- ability to generalize correctly for future data.
This function shows this concept:
![[Pasted image 20251010103425.png]]

# Overfitting

**Definition**: When the predictor has excellent performance on the training set, but its performance on the true "world" is very poor.

Hence, overfitting is not about noise nor about having training error 0, but about the fact that training error and true error are two different things.

They behave similarly only under certain condition, for example, if we have many samples compared to the number of hypotheses.

# Cross validation

Cross validation is a technique that helps us improve the model performance by dividing the input data into two parts:
- **training data**: the data we actually use to train the model;
- **validation data**: the data we use to asses the model performance, in other words, we simulate how the model will generalize future data.

A pretty powerful variant of this technique is **k-fold cross validation**.

## k-fold Cross Validation

We can choose different samples of training data and validation data from all the data we have and run the simulation multiple times each one with a different set of these data.
In the picture below, there's an example of a 4-fold cross validation:
![[Pasted image 20251010111402.png]]
>[!info]
>This technique is super useful when we're working with small samples of data, because it allows us to achieve the highest generalization grade without needing to add more data.

## Train - Validation - Test

In practice, we usually have one pool of examples and we split them into three sets:
- **Training set**: this is the set used to simply train the model; thanks to this training process we are able to make some hypotheses. During the training phase, the model learns its parameters.
- **Validation set**: this is the set used to estimate the generalization error during or after training, allowing for the hyperparameters to be updated accordingly.
- **Test set**: this is the set used to evaluate the model performance. The accuracy on the test data gives us an idea on how well the model is able to generalize for new data.

>[!info]
>The **hyperparameters** are called like that because they are parameters that the model doesn't learn by itself, but they are assessed (by us humans) during the validation process.

>[!warning]
>If the number of samples in the test set is too small, I cannot reliably estimate the true error!

>[!info]
>The thing is the model doesn't have access to the test set. So, in order to tell if the model is overfit or not, we need to check if the discrepancy between the training error and the validation error is high or not.

## Model selection in summary

There is an **unavoidable** trade-off between complexity of the model (which is related to training error) and its variance (which is related to test error).

>[!warning]
>**Never use the test set for any selection of parameters/algorithms.**
>This is because doing so will only give the illusion that the model is performing better, but in reality it's increasing the risk of introducing biases and overfit the model (result: you have a model with good performance with your test data but fails with the actual new data).

## Empirical Risk Minimization (ERM)

The Empirical Risk Minimization (ERM) is basically just the attempt to find a classifier that has the lowest empirical risk among all the other possible classifiers, given some training data $S$ and $m$ samples.

That classifier can be found in this way: $$h_S=\arg \min_{h\in H} \frac1m \sum\limits^m_{i=1}\Bbb 1[h(x_i)\not =y_i]$$
This formula can be also written in this way: $$h_S=\arg \min_{h\in H} R_S(h)$$
## Regularization

Regularization allows us to control the complexity of our models by penalizing complexity.

So, we can adjust the ERM formula to take into account also this parameter, and the result is: $$h_S=\arg \min_{h\in H} (R_S(h)+\lambda\ C(h))$$
>[!info]
>In this case, $C$ stands for "Complexity", referring to the complexity of the model.

λ is a parameter, a positive number that serves as a conversion rate between the loss and the hypothesis complexity (they might not have the same scale).

We still need cross validation and in this case we need to include the parameter λ - select the one that gives the best validation score.

# Linear Regression

So far, we've focused on classification problems where the label space is discrete.

**Regression** is different though. Regression analysis considers prediction problems where label space $y$ is continuous (e.g. $[0, 1]$, $\Bbb{R}^d$, etc.).

Measuring quality of a predictor $h: x\to y$ using prediction error $\Bbb P[h(x)\not =Y]$ doesn't make much sense in the regression context.
>[!info]
>The reason is simple, since we're working with continuous values it's basically impossible to get even one prediction with the same exact value as the correct one. So the prediction error would be pretty high in any case.

So the **goal** is: find $h: x\to y$, from some function class $H$, minimizing error measured using a loss function $L: y\times y\to \Bbb R$, i.e. $$\Bbb E[L(Y, h(x))]$$
## Loss function (in general)

A loss function $L: y\times y\to \Bbb R$ maps decisions to costs: $L(\hat y, y)$ defines the penalty paid for predicting $\hat y$ when the true value is $y$.

**Standard choice for classification**: 0/1 loss.
$$L_{0/1}(\hat y, y)= \begin{cases} 0\quad \hat y = y \\ 1 \quad otherwise \end{cases}$$

**Standard choice for regression**: squared loss.
$$L(\hat y, y) = (\hat y = y)^2$$
>[!warning]
>Squared loss it's obviously not the only possible chance for regression.

## Empirical Loss

We consider a parametric function $h_W(x)$.

The empirical loss of function $y=h_W(x)$ on a set $S$: $$L_S(W) = \frac1N \sum\limits_{i=1}^N L(h_W(x_i), y_i)$$
The goal is to find an ERM with respect to this loss function.

## Linear Predictors in 1d

Both features and labels are real numbers.

Consider predictors of the form: $$\hat y= h_{(w, b)}(x)= w\ x+b$$
Hypothesis class: $H=\{h_{(w,b)}:\ w,b \in \Bbb R\}$
>[!info]
>**Little reminder**: in reality, $w$ is the weight and $b$ is the bias.

## Linear fitting to data

We want to fit a linear function to an observed set of points $x=[x_1, ..., x_N]$ with associated labels $y=[y_1,...,y_N]$.
Once we fit the function, we want to use it to predict the $y$ for new $x$.

![[Pasted image 20251014110644.png | 500]]

One criterion we can use is the **Least Squares (LSQ) fitting criterion**: find the function that minimizes sum (or average) of square distances between actual $y_i$ in the training set and predicted ones, that is ERM.

The result will look like this:
![[Pasted image 20251014110800.png | 500]]

## Linear functions

General form of a linear function: $$\hat y=h_{(b,\mathbf w)}(\mathbf x)= b+w_1x_1+...+w_dx_d=b+\braket{\mathbf w, \mathbf x}$$
Cases:
- 1 dimension ($X = \Bbb R$): a line;
- 2 dimensions: a plane;
	- example: 
	  ![[Pasted image 20251014112520.png | 200]]
- d dimensions: a **hyperplane**.

Hypothesis class (in general): $H = \{h_{(b, \mathbf w)}: b\in \Bbb R, \mathbf w\in \Bbb R^d \}$

## Least Squares Criterion (general case)

Given the hypothesis class $H = \{h_{(b, \mathbf w)}: b\in \Bbb R, \mathbf w\in \Bbb R^d \}$ the squared error on $(x, y)$ is: $$(y-(b+\braket{\mathbf w, \mathbf x}))^2$$
### Least Squares in Matrix/Vector form

>[!note] Personal note
>I didn't understand this part, I give up.

There are some operations but the result is: $$\|y-Xw\|_2^2$$
>[!info]
>This notation means 2 things:
>1. The $\|.\|_2$ notation is the **Euclidean Norm**, which corresponds to this: $$\|v\|_2=\sqrt{v_1^2+v_2^2+...+v_m^2}$$.
>2. The $(.)^2$ notation means you just square the result of the Euclidean norm and, thus, you obtain: $$\|v\|_2^2=v_1^2+v_2^2+...+v_m^2$$.

## Least Squares via Calculus

From calculus point of view, least squares criterion is just a convex function of $\bf w$. So, it is sufficient to find the $\bf w$ where the gradient is zero.

>[!note] Personal note
>Then there are equations that mean nothing to me, they're skipped for now.

## Linear predictor in 1d

If we're trying to find the optimal function, we're looking for: $$\arg\min_w \sum\limits_i(y_i-wx_i)^2$$

And that is found by taking its derivative with respect to $w$ and set it to 0.
So: $$
\begin{gathered}
\frac\partial{\partial w} \sum\limits_i(y_i-wx_i)^2= 2\sum\limits_i-x_i(y_i-wx_i) \implies \\ \ \\
2\sum\limits_ix_i(y_i-wx_i)=0 \implies 2\sum\limits_ix_iy_i- 2\sum\limits_iwx_ix_i=0 \implies \\ \ \\
\sum\limits_ix_iy_i=\sum\limits_iwx_i^2 \implies \\ \ \\
w= \frac{\sum\limits_ix_iy_i}{\sum\limits_ix_i^2}
\end{gathered}
$$
>[!warning]
>We didn't consider the bias for now.

### Add the bias term

So far we assumed that the line passes through the origin.

What if the line does not?
![[Pasted image 20251015090828.png | 350]]

We can then simply change the model to $y=w_0+w_1x$.

And we can still use least squares to determine $w_0$ and $w_1$:
$$w_0=\frac{\sum\limits_iy_i-w_1x_i}n \qquad \qquad w_1= \frac{\sum\limits_i x_i(y_i-w_0)}{\sum\limits_ix_i^2}$$
>[!note] Personal note
>I'm skipping to page 59 because this is too much.

# Use a regression model for classification

What if we want to predict a category instead of a value?

A possible solution is to do **post-processing** to convert linear regression to a binary output.

Therefore, the solution is not necessarily an optimum anymore.

So instead, we'll modify the loss to minimize over categorical values directly.

# Logistic regression

The new loss function is (PLEASE REPLACE THE IMAGE WHENEVER IT'S POSSIBLE):
![[Pasted image 20251015210647.png]]

![[Pasted image 20251015211000.png]]

![[Pasted image 20251015211026.png]]

## What was the loss in SVM?

![[Pasted image 20251015211345.png]]

## Finding a solution

>[!note] Personal note
>I'm skipping all the steps before

For this type of loss, when we're trying to find: $$\nabla_\Theta \ell_\Theta=0 $$
we're working with a non linear system and, therefore, there are no analytical solutions.
>[!info]
>Indeed, the process to find a solution to logistic regression is known as **non linear optimization**.

