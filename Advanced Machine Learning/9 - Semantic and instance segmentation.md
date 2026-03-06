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

>[!todo]
>From page 11 to page 26 we can see how we can perform upsampling. Since I wanna rush I won't make notes on this.

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

>[!todo]
>There are also other CV tasks that I'm skipping for now.
>They are:
>- Panoptic segmentation.
>- Human keypoints.
>- Joint instance segmentation and pose estimation.
>- Dense captioning.

