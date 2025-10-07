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

(page 68)