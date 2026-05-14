# Semantic segmentation

Semantic segmentation consists in labeling each pixel in the image with a category label.
It doesn’t differentiate instances, only care about pixels.

>[!note]
>This means that segmentation doesn't consider each object individually (for example, it doesn't make the distinction between the 2 cows, it just says "there's a cow area").

![[Pasted image 20260306154943.png]]

## Semantic segmentation ideas

### Sliding window

Basically just analyzing one region of the image at a time.
![[Pasted image 20260306155308.png]]

**Problem**: this approach is very inefficient, since it doesn't reuse shared features between overlapping patches.

### Fully convolutional

Design a network as a bunch of convolutional layers to make predictions for pixels all at once!

![[Pasted image 20260306155530.png]]

This approach has two **problems**:
- Effective receptive field size is linear in number of conv layers → many layers are required.
- Convolution on high res images is expensive.

So, to reduce costs (and waste of resources) **downsampling** and **upsampling** are performed:
![[Pasted image 20260306160044.png]]

**Problem**: we know how to manage downsampling but not how to manage upsampling.


### In-network upsampling
#### "unpooling"

>[!Question]
>This is asked in the exam maybe.
>Does pooling have learnable parameters? The answer is no.
>Does unpooling have learnable parameters? The answer is again no.

There are two approaches to perform unpooling:
![[Pasted image 20260430152806.png]]

#### Bilinear interpolation

![[Pasted image 20260430153045.png]]

#### Bicubic interpolation

![[Pasted image 20260430153120.png]]

#### "Max unpooling"

![[Pasted image 20260430153222.png]]

### Learnable upsampling: transpose convolution

![[Pasted image 20260430154043.png]]

![[Pasted image 20260430154059.png]]

#### Transpose convolution: 1D example

![[Pasted image 20260430154145.png]]

Other names:
- Deconvolution (bad)
- Upconvolution
- Fractionally strided convolution
- Backward strided convolution

# Computer vision tasks
![[Pasted image 20260306160428.png]]

# Things and stuff
![[Pasted image 20260306160449.png]]

# Instance segmentation
![[Pasted image 20260306160631.png]]

>[!note]
>Basically it uses Mask R-CNN networks, which are Faster R-CNN networks (already encountered in object detection) with mask prediction.
>Here's an image about it:
>![[Pasted image 20260306161252.png]]
## Beyond instance segmentation
![[Pasted image 20260306161343.png]]

## Panoptic segmentation

It labels all the pixels in the image (both things and stuff), but also differentiates between the instances of the things. It's basically a fusion of semantic segmentation and instance segmentation.

![[Pasted image 20260430161907.png]]

>[!note] Personal note
>There are also other CV tasks that I'm skipping for now.
>They are:
>- Human keypoints.
>- Joint instance segmentation and pose estimation.
>- Dense captioning.

