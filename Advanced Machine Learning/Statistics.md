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

## Bayes' rule

Formula: $$ P(x|y) = \frac{P(y|x)}{P(y)} P(x) $$
As we'll discover later, this rule allows us to calculate a (maybe) difficult conditional probability using an easier conditional probability.
> Sometimes inverting cause and effect can make calculations easier.

(CONTINUE FROM PAGE 44)