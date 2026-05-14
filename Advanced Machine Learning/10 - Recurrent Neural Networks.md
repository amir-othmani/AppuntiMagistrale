This kind of network is used for **sequential data**:
- Videos
- Time series data
- Speech / Music
- User behaviour in websites

The typical applications are:
- Machine Translation
- Image Captioning
- Question Answering
- Video Generation
- Speech Synthesis
- Speech Recognition

# "Vanilla" Neural Network
There are many types:
- ![[Pasted image 20260310211242.png | 500]]
- ![[Pasted image 20260310211349.png | 500]]
- ![[Pasted image 20260310211419.png | 500]]
- ![[Pasted image 20260310211442.png | 500]]
- ![[Pasted image 20260310211519.png]]
	- Video classification on frame level:
		- sequence of images → sequence of labels

# Recurrent Neural Networks
Every RNN has an "internal state" that is updated as a sequence is processed:
![[Pasted image 20260310212024.png]]

Unrolled RNN:
![[Pasted image 20260310212054.png]]

We can process a sequence of vectors x by applying a recurrence formula at every time step:
![[Pasted image 20260310212530.png]]

The general mechanism of an RNN is something like this:
![[Pasted image 20260310212826.png]]

>[!note]
>Pay attention to how loss is calculated for each RNN neuron and the final loss is a result of those intermediate losses.

## Backpropagation through time
Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient:
![[Pasted image 20260310212945.png]]

## Truncated Backpropagation through time
Run forward and backward through chunks of the sequence instead of whole sequence:
![[Pasted image 20260310213119.png]]

## RNN tradeoffs
RNN Advantages:
- Can process any length of the input (no context length)
- Computation for step t can (in theory) use information from many steps back
- Model size does not increase for longer input
- The same weights are applied on every timestep, so there is symmetry in how inputs are processed
RNN Disadvantages:
- Recurrent computation is slow
- In practice, difficult to access information from many steps back

# Image captioning

Image captioning is a task where the model is trying to provide a description from an image provided as an input.

![[Pasted image 20260430173243.png]]

# Vanilla RNN gradient flow

A single step works like this (general form):
![[Pasted image 20260430174426.png]]

And, when we put the steps together, the result is something like this:
![[Pasted image 20260430174542.png]]

Depending on the gradient value it might happen:
- **Exploding gradients**, if the largest singular value is >1.
- **Vanishing gradients**, if the largest singular value is <1.

To address exploding gradients, we can perform **gradient clipping**, which is scaling the gradient if the norm is too big.

Example code:
````python
grad_norm = np.sum(grad * grad)
if grad_norm > threshold:
	grad *= (threshold / grad_norm)
````

To address vanishing gradients there's no easy solution, it is necessary to change the RNN architecture.

# Long Short Term Memory (LSTM)

![[Pasted image 20260430175854.png]]

LSTM introduces other non-linearities (with sigmoids) and new hidden states called **cell states** (the ct in the formula).

There are multiple types of gates:
![[Pasted image 20260430180359.png]]

There's also this picture which I don't really know what to say about:
![[Pasted image 20260430180508.png]]

This is an example of single layer RNN with LSTM:
![[Pasted image 20260430180629.png]]

But there can be more than one hidden layer, like this:
![[Pasted image 20260430180712.png]]

# Take home messages about this chapter
![[Pasted image 20260430180759.png]]

# Sequence-to-Sequence with RNNs

Example with a translation task:
![[Pasted image 20260501110542.png]]

In this case:
- s0 is the **initial decoder state**, which is the output of the encoder and it's the result of the elaboration of the input states and the hidden states;
- c is the **context vector** and it's usually just a copy of the last hidden state.

>[!warning] Problem
>This architecture could be fine if the sequence length (T) is short, but if the sequence length is very long, the context vector might act as a bottleneck for the decoder.

>[!todo] Solution
>A solution could be using a new context vector at each step of the decoder.

## Sequence-to-Sequence with RNNs and Attention

![[Pasted image 20260501112041.png]]

Basically we compute some **alignment scores**, which tell us how much the hidden states are aligned with the final state s0 (usually the computation is performed with a dot product).

![[Pasted image 20260501112217.png]]

Then we can perform a softmax to normalize the alignment scores, which become **attention weights**, so that they sum up to 1 (each attention weight has to be between 0 and 1).

So, each attention weight provides an indication of how much the corresponding word is relevant and this will be important when we compute the context vector.

![[Pasted image 20260501113804.png]]

In this way, the context vector won't heavily depend on the last hidden state (and therefore, the last word) but it will main depend on the most relevant word (or the most relevant words if there's a tie).

But we're not done yet, because then we will compute another context vector to obtain the next decoder state:
![[Pasted image 20260501113834.png]]

As we can see, s1 is used to compute the relation with the hidden states, and so new attention weights will be computed (these new attention weights have to be different compared to the previous ones, because we're trying to predict the next translated word).

So, this process is iterated until the end of the sentence:
![[Pasted image 20260501114924.png]]

A visual example of the attention weights:
![[Pasted image 20260501115140.png]]

## Image captioning with RNNs and Attention

It is possible to re-use the RNN+Attention mechanism for image captioning.
![[Pasted image 20260501131749.png]]
![[Pasted image 20260501131826.png]]
![[Pasted image 20260501131842.png]]
>[!note]
>Each timestep of the decoder uses a different context vector that looks at different parts of the input image.

