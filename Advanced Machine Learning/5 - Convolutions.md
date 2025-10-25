# Neural network

Any neural network looks like this:
![[Pasted image 20251020204744.png]]

With:
- **Output**: $g^{out}=\sigma (\dots W^{(2)}\sigma (W^{(1)} f^{in}))$
- **Activation**: e.g. ReLU $\sigma=\max\{x, 0\}$.
- **Parameters**: weight of all layers $W^{(1)},\dots,W^{(L)}$ (including biases).

# Priors

## The need for priors

Deep feed-forward networks are probably **universal**.

>[!info]
>"feed-forward" implies there's only one direction-flow in the neural network.

However:
- We can make them **arbitrarily complex**.
- The number of **parameters** can be huge.
- They're very difficult to **optimize**.
- It's very difficult to achieve **generalization**.

>[!question]
>What do you mean "It's very difficult to achieve **generalization**"? Aren't those neural networks universal???

In order to face these problems, we need additional **priors** (e.g., some additional prior info).

Those priors should be as much universal as possible and have to be **task-independent** in some way.
Task-independent priors must come with data.

## Structure as a strong prior

Usually data have in themselves some structural priors related to repeated patterns, compositionality, locality and so on.

Some of those priors can be:
- Self-similarity.
- Translation invariance.
- Hierarchy and compositionality.

# Convolutional neural networks

CNNs are a type of neural network that are excellent when the data is composed of hierarchical, local and shift-invariant patterns, since they take those as priors.

## Sparse interaction

Comparison between fully-connected layer and convolutional layer:
![[Pasted image 20251020213418.png]]

# Convolution

Given two functions $f,g: [-\pi, \pi] \to \Bbb R$, their convolution is a function:
$$ (f\star g)(x)=\int\limits_{-\pi}^\pi f(t)g(x-t)dt $$
where:
- $(f\star g)(x)$ is called **feature map**.
- $g(x-t)$ is called **kernel**.

Example where we use the same f as kernel:
![[Pasted image 20251020213856.png]]

Example where we use a generic function (g) as kernel:
![[Pasted image 20251020213925.png]]

## Properties

The convolution is **commutative**:
$$ (f\star g)(x)=(g\star f)(x) $$
and it's also **shift-equivalent**:
$$ f(x-x_0)\star g(x)=(f\star g)(x-x_0) $$

![[Pasted image 20251021214856.png | 450]]

We can see convolution as the application of a **linear** operator $\mathcal G$:
$$ \mathcal Gf(x)=(f\star g)(x)=\int^\pi_{-\pi}f(t)g(x-t)dt $$
$\mathcal G$ is linear, so:
$$ \mathcal G(\alpha f(x))=\alpha \mathcal Gf(x) $$
$$ \mathcal G(f+h)(x)=\mathcal Gf(x)+\mathcal Gh(x) $$

So, translation equivariance can be phrased as:
$$ \mathcal G(\mathcal Tf)=\mathcal T(\mathcal Gf) $$

## Discrete convolutions

In a discrete domain the convolution is defined as a **convolution sum**:
$$ (f\star g)[n]=\sum\limits^\infty_{k=-\infty} f[k]g[n-k] $$

# Convolutional neural network

>[!note] Personal note
>I don't know what to write on this part, so here's the screenshot:
>![[Pasted image 20251021220815.png]]
>Although, there's some pooling between each layer and the other:
>![[Pasted image 20251021221125.png]]

## Local filters

>[!note] Personal note
>Read the slides for this part and what's next, it's better to jump directly on next chapter.
>PS: I changed my mind, I'll write something but very minimal.

Shift-invariance is implemented via **convolutional operators**.

The complexity in this technique is $O(1)$, which is a huge gain compared to MLP.

Filter weights are applied across the entire image, this is known as **weight sharing**.


## Pooling

Pooling is just a technique in which we reduce the amount of data by applying some operations to the original data (could be max, min, sum, average or whatever).

Example:
![[Pasted image 20251022090433.png]]

## Key properties of CNN

- Convolutional filters **(translation equivariance)**.
- Multiple layers **(compositionality)**
- Filters localized in space **(locality)**
- Weight sharing **(self-similarity)**
- $O(1)$ parameters per filter (independent of image size $n$).

