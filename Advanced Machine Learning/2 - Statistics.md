# Basic statistics

We generally want to use machine learning for one of these goals:
- **diagnosys**
- **anomaly detection**
- **reinforcement learning**
- **recommender system**.

All of these goals are hindered by **uncertainty**, more specifically:
- uncertain **input**: missing or noisy data;
- uncertain **knowledge**: we never have full knowledge of the real world;
- uncertain **outputs**: 
	- induction is inherently uncertain, and
	- deduction may be uncertain as well if incomplete.

## Sample spaces

A **sample space** Ω is the set of all possible outcomes of a random experiment. (Ω can be finite or infinite.)

Examples:
- Rolling a dice: {1,2,3,4,5,6}
- Flipping a coin: {H, T}
- Flipping a coin three times: {HHH, HHT, HTH, HTT, THH, THT, TTH, TTT}
- A person’s age: the positive integers
- A person’s height: the positive reals.

## Event

An **event** is a subset of the sample space Ω.

## Probability

**Probability** is a function that maps an event onto the interval [0, 1].

### The axioms of probability

1. All probabilities are between 0 and 1, thus: $0 ≤ P(A) ≤ 1$
2. **Valid** propositions have probability 1,
   **Unsatisfiable** propositions have probability 0.
   P(empty-set) = 0, P(everything) = 1.
3. The probability of a disjunction is given by: $$P(A \cup B) = P(A) + P(B) – P(A \cap B)$$
## Random variables

A **random variable** is a function of the outcome of a randomized experiment. In other words, if we have a sample space, we can apply a function on each event on it and obtain a new output.

For example, we can consider throwing two dice (with 4 faces) as an event and the sum of their result as the random variable.

|                              |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Throwing  2 dice (event)** | 1-1 | 1-2 | 1-3 | 1-4 | 2-1 | 2-2 | 2-3 | 2-4 | 3-1 | 3-2 | 3-3 | 3-4 | 4-1 | 4-2 | 4-3 | 4-4 |
| **Sum (random variable)**    | 2   | 3   | 4   | 5   | 3   | 4   | 5   | 6   | 4   | 5   | 6   | 7   | 5   | 6   | 7   | 8   |


We can also calculate the **probability of the random variable**.

|            |                    |                    |                    |                    |                    |                    |                    |
| ---------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **Sum**    | 2                  | 3                  | 4                  | 5                  | 6                  | 7                  | 8                  |
| **P(sum)** | $\Large\frac1{16}$ | $\Large\frac2{16}$ | $\Large\frac3{16}$ | $\Large\frac4{16}$ | $\Large\frac3{16}$ | $\Large\frac2{16}$ | $\Large\frac1{16}$ |
## Probability distribution

A **distribution** is a table of probability values.
Example:
![[Pasted image 20251005214736.png| 550]]

### Joint probability distribution

We can measure the distribution of two events combined.

Example:
![[Pasted image 20251005215802.png]]

### Marginal distribution

**Marginal distributions** are sub-tables which eliminate variables.
Marginalization is done by collapsing rows by addition.

Example:
![[Pasted image 20251005215958.png]]

## Conditional probabilities

P(X|Y) = Fraction of worlds in which X event is true given Y event is true.

Formula: $$ P(a|b) = \frac{P(a,b)}{P(b)} $$
Conditional distribution is just a marginal distribution at the end of the day.
>[!info]
> NB: $P(a,b)$ is equivalent to $P(a \cap b)$.

## Bayes' rule

Formula: $$ P(x|y) = \frac{P(y|x)}{P(y)} P(x) $$
As we'll discover later, this rule allows us to calculate a (maybe) difficult conditional probability using an easier conditional probability.
>[!info]
> Sometimes inverting cause and effect can make calculations easier.

### Bayes' rule important implications

In certain conditions, the Bayes' rule has also this form: $$P(x|y) \propto P(y|x)P(x)$$
where:
- **x** is the **hypothesis**;
- **y** is the **evidence**.

In this context, all of the three probabilities have specific names:
- $P(x|y)$ is called **posterior** probability.
	- In other words, this is the probability that a certain cause happened given a certain effect we could observe.
- $P(y|x)$ is called **likelihood**.
	- In other words, this is the probability that a certain effect will happen given a certain cause.
- $P(x)$ is called **prior** probability.
	- In other words, this is the probability that a certain cause might happen in general.

## Independence

Two events are called **independent** if they don't contain information about each other.
For instance, if we consider two events X and Y, they are independent if X doesn't affect the probability of Y to happen and vice versa (example: throwing two coins).

If two events are independent these equations are valid: $$ P(X,Y) = P(X)P(Y)$$ $$P(X|Y) = P(X) $$
Examples:
- **Independent**: Winning on roulette this week and next week.
- **Dependent**: Russian roulette.

### Conditionally independent

Two events are **conditionally independent** if knowing that a third event happened makes the first two independent.
For instance: $$P(X,Y|Z) = P(X|Z)P(Y|Z)$$
which means that if we know Z happened, then X and Y are independent (i.e., X doesn't affect Y probability and vice versa).

## Model based classification with Naive Bayes

Let's say we have a $X = (X_1 \ X_2\ ...\ X_d)$ dataset and we want to predict the class label of $Y$; in other words, we want to find the function $y=f(x)$.

Example:
![[Pasted image 20251006091304.png]]

In order to make that prediction we can use Bayes formula: $$ P(x|y) = \frac{P(y|x)}{P(y)} P(x) $$
Generally speaking:
- Estimating $P(y)$ is easy.
- Estimating $P(x|y)$ , however, is not easy!
	- To make it easier, we can assume that all the attributes (e.g., the $X_i$ events) are independent given the class label $Y$ (e.g., **conditionally independent**).

Naive Bayes classifier properties:
- It's computationally very fast
	- Training: only one pass over the training set.
	- Classification: linear in the number of attributes (features).
- Despite its conditional independence assumption, Naïve Bayes classifier shows a good performance in several application domains.
- When to use?
	- A moderate or large training set available with instances represented by a large number of attributes.

## Laplace smoothing

It might happen that an attribute never appears for a specific class. That can cause problems because some events would look impossible or certain even if it might not be true (let's say it's extremely unlikely to happen).

For example:
![[Pasted image 20251006093047.png]]

So, a simple solution is adding the missing attribute(s) (e.g., adding another row to the table) so we don't have anymore probabilities equal to 0 or 1.

## Discriminative vs Generative learning

Many supervised learning can be viewed as estimating $P(X,Y)$. Generally thy fall into two categories:
- When we estimate $P(X,Y)=P(X|Y)P(Y)$ then we call it generative learning.
- When we only estimate $P(Y|X)$ then we call it discriminative learning.
![[Pasted image 20251006093732.png]]

## Maximum a Posteriori (MAP) & Bayesian Learning

When we perform **Bayesian learning**, we are trying to evaluate the probability of an unknown quantity, given some observed data, by using a set of hypothesis.
In other words, we're performing this calculation: $$P(X|d) = \sum\limits_i P(X|h_i)P(h_i|d)$$
where:
- **b** is the observed data;
- **X** is the unknown quantity;
- $\bf{h_i}$ is one of the hypotheses.

So, the probability we're looking for is just a linear combination of the probabilities for each hypothesis.

If we decide to perform **Maximum a Posteriori (MAP)**, we perform an approximation of that probability by taking into account only $h_{MAP}$, which is the hypothesis that has the highest probability among the others.
Therefore: $$P(X|d) \approx P(X|h_{MAP})$$
>[!info]
>This approximation is justified by Naive Bayes and the fact that the probabilities taken into account are conditionally independent.

## Independent and Identically distributed (i.i.d.)

This is an assumption we may make where:
1. The events are **independent from each other**, thus: $$P(E_j|E_{j-1}, E_{j-2},...) = P(E_j)$$
2. The events are **identically distributed**, meaning that they have the same probability, so: $$P(E_j)= P(E_{j-1})=P(E_{j_2})=...$$
The iid assumption connects the past to the future, without some such connections, all the bets are off and the future could be anything.
## Maximum A Posteriori → Maximum Likelihood (ML)

Given this formula: $$P(y|x) \propto P(x|y)P(y)$$
>[!warning]
>Careful, this time **y** is the hypothesis and **x** is the evidence.

The prior can be used to penalize complexity, since more complex hypotheses will have lower probability.
Anyway, if we consider hypotheses that have all the same probability, then the maximum posterior is found with the **maximum likelihood**.
This approach provides a good approximation to Bayesian and MAP learning when the data set is large.
## Continuous Random Variables

A random variable X is continuous if its set of possible values is an entire interval of numbers.

Example:
![[Pasted image 20251007120136.png]]

>[!warning]
>**Remember**:
>- The **x** axis represents the possible values.
>- The **y** axis represents the frequency for each value.

>[!personal note]
>I'm skipping the rest

## How good is the estimator?

Consider two properties of the estimator:
- **Bias**, which is the distance between the estimate and the true value.
- **Variance**, which indicates how many different values there are around the mean.

![[Pasted image 20251007122312.png]]

## The bias

The bias for an estimator $\hat \theta_m$ for parameter $\theta$ is defined as: $$bias(\hat\theta_m) = E[\hat\theta_m]-\theta$$
where $E$ stands for estimated(?).
The estimator is unbiased if $bias(\hat\theta_m) = 0$.

## The variance

The variance indicates how much we expect the estimator to vary as a function of data samples.
The variance of an estimator is simply Var($\hat\theta$) where the random variable is the training set.
If two estimators of a parameters are both unbiased, the best is the one with the least amount of variability (because it's more efficient).

The square root of the variance is called the **standard error**, denoted as SE($\hat\theta$).
 SE($\hat\theta$) measures how we would expect the estimate to vary as we obtained  different samples from the same distribution.
## Mean Squared Error

Let $\hat\theta$ be the estimator for an unknown parameter $\theta$. The **Mean Squared Error (MSE)** is defined as: $$MSE(\hat\theta)=E[(\hat\theta-\theta)^2]$$
There's also the **Mean Absolute Error**, which is: $$MAE(\hat\theta)=E[|\hat\theta-\theta|]$$
