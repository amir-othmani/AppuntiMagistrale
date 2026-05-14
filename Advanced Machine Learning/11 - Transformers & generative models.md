# Attention

The attention layer has this structure:
![[Pasted image 20260505103532.png]]

>[!note]
>The basic attention layer is already covered in the previous chapter.

>[!question]
>How can we improve this architecture?

## Similarities with dot product

Dot product is the simplest way to compute similiraties and its formula is:
$$ e[N_x]:\quad e_i = q\cdot X_i $$

There's also a **scaled** variant:
$$ e[N_x]:\quad e_i = \frac{q\cdot X_i}{\sqrt{D_X}} $$
This one is used if there are too large similarities. Large similarities cause softmax to saturate and give vanishing gradients.

## Multiple query vectors

![[Pasted image 20260505105518.png]]

>[!note] Personal note
> Look, the professor didn't explain very clearly what are the benefits of using multiple query vectors, so this paragraph will remain without any explanations.

## Separate key and value

![[Pasted image 20260505110524.png]]

>[!note] Personal note
>From what I understood, separating key and value is useful to analyze more accurately the input.

## Cross-attention layer

The result is this architecture:
![[Pasted image 20260505112416.png]]

In cross-attention:
- Each query produces one output, which is a mix of information in the data vectors.
## Self-attention layer

There's also this one:
![[Pasted image 20260505112034.png]]

It's called "self-attention" because the queries aren't external but they are generated from the input data.

In self-attention:
- Each input produces one output, which is a mix of information from all inputs.

Steps:
1. ![[Pasted image 20260505113606.png]]
2. ![[Pasted image 20260505113652.png]]
3. ![[Pasted image 20260505113702.png]]
4. ![[Pasted image 20260505113741.png]]

Now consider **permuting inputs**:
- Queries, keys and values will be the same but permuted.
- Similarities are the same but permuted.
- Same for attention weights.
- Same for the outputs.

Result:
![[Pasted image 20260505113921.png]]

>[!note]
>All of this is possible because self-attention is **permutation equivariant**.

![[Pasted image 20260505114103.png]]

## Masked self-attention layer

![[Pasted image 20260505114210.png]]

## Multiheaded self-attention layer

![[Pasted image 20260505114303.png]]

## Three ways of processing sequences
![[Pasted image 20260505115511.png]]

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

# Explicit density estimation

Let's say we have a dataset with a few samples and we don't know any information about the distribution those samples come from.

So, we can use **explicit density estimation** to estimate that original distribution:
![[Pasted image 20260513175556.png]]

>[!abstract] Explanation
>W* is the (best) estimated distribution that our model can find from the dataset. W* is obtained with the product of the maximum likelihood of each data sample; but that result can be obtained also with the log likelihood, which is convenient because then we have a sum instead of a product. 
>That log likelihood corresponds to the loss function obtained with gradient descent.

It is also possible to compute that estimation by dividing each samples in subparts:
![[Pasted image 20260513182105.png]]

## Autoregressive models of images

A way to generate images is to treat an image as a sequence of pixels (8-bit).
This can be done with RNNs or Transformers. In either case, the model would create the image by producing one pixel at a time.

>[!danger] Problem
>This process is really expensive. For example, a 1024x1024 would require a sequence of 3M pixels.

>[!success] Solution
>Instead of processing single pixels, group them into tiles and then model the image as a sequence of those tiles.

# Autoencoders
Autoencoders have this architecture:
![[Pasted image 20260316091859.png]]

It is an unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data.

>[!note]
>z is usually smaller than x (dimensionality reduction).

![[Pasted image 20260316093136.png]]

![[Pasted image 20260316093219.png]]

Autoencoders can reconstruct data, and can learn features to initialize a supervised model.

>[!error] Problem
>Generating new z is not any easier than generating new x.

>[!success] Solution
>What if we force all z to come from a known distribution?

# Variational autoencoders
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

Thus, the solution is to approximate $p_\theta(x|z)$ with an additional encoder network $q_\phi(z|x)$.

![[Pasted image 20260316095642.png]]

![[Pasted image 20260514123549.png]]

![[Pasted image 20260316095714.png]]

# Generative Adversiarial Networks

>[!tip] The idea
>Instead of using explicit density to model p(x) (which is intractable), we use a game theory approach: learn o generate from training distribution through 2-player game. This will allow us to sample from p(x).

>[!error] Problem
>Want to sample from complex, high-dimensional training distribution. But there is no direct way to do this!

>[!success] Solution
>Sample from a simple distribution, e.g. random noise. Learn transformation to training distribution.

>[!question]
>What can we use to represent this complex transformation?

>[!todo] Answer
>We can take in input some random noise and generate a distribution with a Generator Network.

![[Pasted image 20260514153727.png]]

## Training GANs: two player game

**Generator network**: try to fool the discriminator by generating real-looking images.
**Discriminator network**: try to distinguish between real and fake images.

![[Pasted image 20260514155134.png]]

With this mechanism the discriminator network and the generator network help each other to improve themselves:
1. the generator network produces fake images that can fool the discriminator, so the discriminator has to improve in order to be able to distinguish fake images from real ones;
2. if the discriminator network is good at distinguishing fake images from real ones, the generator network has to improve in order to be able to fool the discriminator.

Those two networks are trained in a **minmax game**:
![[Pasted image 20260514160308.png]]where:
- Discriminator ($θ_d$) wants to maximize objective such that D(x) is close to 1 (real) and D(G(z)) is close to 0 (fake).
- Generator ($θ_g$) wants to minimize objective such that D(G(z)) is close to 1 (discriminator is fooled into thinking generated G(z) is real).

![[Pasted image 20260514160459.png]]

![[Pasted image 20260514163207.png]]

>[!note] Personal note
>I'm skipping a little bit because I really wanna finish this shit.

# Diffusion models

>[!note] Personal note
>Look, I'm very fed up with this shit of a subject and I'm gonna study this section directly from the holy notebook.

