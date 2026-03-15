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

>[!todo]
>(from page 66 to the end)
>I'll come back to this part later

