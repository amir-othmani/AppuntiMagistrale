>[!todo]
>Read the slides for the attention part (which is from the start until page 40).

# The transformer
![[Pasted image 20260311210630.png]]![[Pasted image 20260311210707.png]]

## Transformers for Language Modelling (LLM)
![[Pasted image 20260311210803.png]]

# Supervised vs Unsupervised Learning
![[Pasted image 20260315201911.png]]

## Self-supervision
It's not always possible to have enough labels and/or annotations to train properly a model.
In this case, we can use a form of unsupervised learning where the data provides the supervision. This is known as **self-supervision**.

**Standard approach**:
1. use self-supervision as a pretext to learn features
2. transfer the features to a task with a limited amount of samples

![[Pasted image 20260315210912.png]]

# Generative vs discriminative models
![[Pasted image 20260315211108.png]]

## Generative models
They are used when many outputs x are possible for an input y and so we instead map the probability P(x|y).

>[!note]
>Examples are LLMs, which are capable to produce a text or an image from a prompt.

## Taxonomy of generative models
![[Pasted image 20260315211539.png]]

>[!todo]
>I'm skipping explicit density estimation.

# Autoencoders
Autoencoders have this architecture:
![[Pasted image 20260316091859.png]]

It is an unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data.

>[!note]
>z is usually smaller than x (dimensionality reduction).

![[Pasted image 20260316093136.png]]

>[!note] Personal note
>These slides are so garbage omg.

![[Pasted image 20260316093219.png]]

Autoencoders can reconstruct data, and can learn features to initialize a supervised model.

## Variational autoencoders
Probabilistic spin on autoencoders - will let us sample from the model to generate data!
![[Pasted image 20260316093803.png]]

Intuition (remember from autoencoders!): x is an image, z is latent factors used to
generate x: attributes, orientation, etc.

We want to estimate the true parameters θ* of this generative model.

How should we represent this model?

Choose prior p(z) to be simple, e.g. Gaussian. Reasonable for latent attributes, e.g. pose, how much smile.
Conditional p(x|z) is complex (generates image) => represent with neural network.

Learn model parameters to maximize likelihood of training data:
$$ p_\theta(x) = \int p_\theta(z)p_\theta(x|z)dz $$

![[Pasted image 20260316095354.png]]

This is not feasible to compute for every z.

Thus, the solution is to approximate $p_\theta(x|z)$ with an additional encoder network $q\_phi(z|x)$.

![[Pasted image 20260316095642.png]]

![[Pasted image 20260316095714.png]]

>[!todo]
>I give up, fuck this shit.

