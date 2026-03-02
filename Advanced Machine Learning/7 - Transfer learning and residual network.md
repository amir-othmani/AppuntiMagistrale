# Models ensembles

Train multiple independent models and, at test time, average their results.

In this way you achieved 2% extra performance.

## Tips and tricks

- Instead of training independent models, use multiple snapshots of a single model during training.
- Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time (Polyak averaging).

# Transfer learning

The core idea of this technique is to make the model use previously learned knowledge even for a new task, instead of making it learn everything from scratch again.

## Transfer learning with CNN

![[Pasted image 20251025113340.png]]![[Pasted image 20251025113702.png]]

>[!info]
>Transfer learning effectiveness changes with different architectures.

>[!info]
>Although, for CNN models it's usually way better to use transfer learning rather then re-learning everything from scratch.

>[!note] Personal note
>There's a comparison between different architectures, but I'm skipping it for now.

## Residual networks

Once we have Batch Normalization, we can train networks with 10+ layers.
What happens as we go deeper?

![[Pasted image 20251025142056.png]]

We can see the deeper model does worse than the shallow model.

Our initial guess is that the deep model is overfitting since it is much bigger than the other model.

Although, this is false because the error rate is high even on the training set:
![[Pasted image 20251025142346.png]]

In fact, the deeper model is actually **underfitting**.

However, deeper models should be able to perform at least as good as shallow models.

>[!question] Hypothesis
>This is an optimization problem. Deeper models are hard to optimize, and in particular don't learn identity functions to emulate shallow models.

>[!success] Solution
>Change the network so learning identity functions with extra layers is easy!

A way to do that is to add an additive "shortcut".
![[Pasted image 20251025143512.png]]

A residual network is a stack of many residual blocks.

Regular design, like VGG: each residual block has two 3x3 conv.

Network is divided into **stages**: the first block of each stage halves the resolution (with stride-2 conv) and doubles the number of channels.

![[Pasted image 20251025144114.png || 550]]

## Basic block

This is an example of basic block:
![[Pasted image 20251025144224.png || 550]]

## Bottleneck block

![[Pasted image 20251025144417.png]]

>[!note] Personal note
>I'm skipping performance and complexity comparison.

# Which architecture should I use?

Don't reinvent the wheel, look for an already public and well-documented architecture/model.

If you just care about accuracy, **ResNet-50** or **ResNet-101** are great choices.

If you want an efficient network (real-time, run on mobile, etc) try **MobileNets** and **ShuffleNets**.

