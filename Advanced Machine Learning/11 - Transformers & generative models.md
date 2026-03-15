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
(page 80)