# Computer Vision tasks

So far we considered only image classification, but there are other tasks (as shown in the following image).

![[Pasted image 20251025145905.png]]

# Object detection

**Input**: single RGB image.

**Output**: a set of detected objects.

![[Pasted image 20251025150938.png]]

For each object predict:
1. Category label (from fixed, known set of categories).
2. Bounding box (four numbers: x, y, width, height).

## Challenges

- **Multiple outputs**: need to output variable numbers of objects per image.
- **Multiple types of output**: need to predict "what" (category label) as well as "where" (bounding box).
- **Large images**: classification works at 224x224; need higher resolution for detection, often ~800x600.

## Multi-Task Learning

- MTL biases the model to prefer representations that other tasks also prefer.
- This will also help the model to generalize to new tasks in the future as a hypothesis space that performs well for a sufficiently large number of training tasks will also perform well for learning novel tasks
- Acts as regularization
- No more one-directional transfer but a symmetric two-directional transfer.

## Detecting a single object

![[Pasted image 20251025155055.png]]
## Detecting multiple objects

![[Pasted image 20251025155142.png]]

## Object Detection as Classification (Sliding Windows)

The idea is to apply a CNN to many different crops of the image where each crop is classified as object or background.

A quick example:
![[Pasted image 20260305122934.png]]
![[Pasted image 20260305122958.png]]

## Region-Based CNN (R-CNN)

The idea is to identify some regions in the image that are likely to contain objects.

It's easier to take a screenshot in this case:
![[Pasted image 20260305123935.png]]

And then the testing session works in this way:
1. Run region proposal method to compute ~2000 region proposals.
2. Resize each region to 224x224 and run independently through CNN to predict class scores and bbox transform.
3. Use scores to select a subset of region proposals to output.
4. Compare ground-truth boxes.

## Comparing boxes: Intersection over Union (IoU)

IoU, as the name says, consists in comparing the intersection between the prediction and the ground-truth boxes with their union:
![[Pasted image 20260305144915.png]]

It is computed with this formula: $$\frac{Area\ of\ intersection }{Area\ of\ Union}$$
and its value can obviously go from 0 to 1.

>[!note]
>Roughly speaking:
>- IoU > 0.5 is "decent"
>- IoU > 0.7 is "pretty good"
>- IoU > 0.9 is "almost perfect"

### Overlapping boxes

If some boxes overlap it becomes difficult to detect objects and computing the IoU becomes tricky.
![[Pasted image 20260305145549.png]]

A way to address this issue is to use **Non-Max Suppression (NMS)**:
1. Select the (next) highest-scoring box
2. Eliminate boxes with IoU lower than a certain threshold (e.g. 0.7)
3. If any boxes remain, go to step 1

>[!warning]
>In this way, if there are many overlapping objects, NMS may eliminate also "good" boxes.


## Evaluating Object Detectors: Mean Average Precision (mAP)

1. Run object detector on all test images (with NMS)
2. For each category, compute Average Precision (AP) = area under Precision vs Recall Curve
	1. For each detection (highest score to lowest score)
		1. If it matches some GT box with IoU > 0.5, mark it as positive and eliminate the GT
		2. Otherwise mark it as negative
		3. Plot a point on PR curve
	2. Average Precision (AP) = area under PR curve
3. Mean Average Precision (mAP) = average of AP for each category
![[Pasted image 20260305152930.png]]

# R-CNN
## Region-Based CNN (basic)

The basic R-CNN works like this:
![[Pasted image 20260428113720.png]]

Where basically a lot of Regions of Interest (RoI) are proposed and each of them is run through a convolutional network.

>[!warning] Problem
>This process is very slow because we need to do ~2k forward passes for each image.

>[!todo] Solution
>A possible solution is to simply run the CNN first and then crop the interested regions.

## Fast R-CNN

It's a variant of R-CNN designed to be faster than regular R-CNN. The idea is to run CNN before warping.

>[!note]
>**Warping** just means that the image region is cropped (or preprocessed in some way) to fit a certain dimension, which was 224x224 in the example.

The approach is shown in this picture:
![[Pasted image 20260306150042.png]]

>[!question]
>How the features are cropped?

### Cropping features: RoI Pool

First of all, we have the image features extracted with the CNN:
![[Pasted image 20260428151011.png]]

Then region proposal is performed on the features:
![[Pasted image 20260428151443.png]]

There is a problem though: the region will probably not be aligned with the feature matrix.

So we have to perform an alignment of the proposed region, but we must keep in mind that this is an approximation and so the region in the matrix doesn't match exactly with the region in the original image.
![[Pasted image 20260428152001.png]]

Then, after we found our region, we need to perform pooling and divide it into multiple equal sub-region.
Although, it's not always possible to divide into perfectly equal sub-regions, so we often will divide the region into roughly equal sub-regions.
Example:
![[Pasted image 20260428152923.png]]

And then, we just have to perform pooling:
![[Pasted image 20260428153022.png]]

There is also an approach that guarantees that the sub-regions will always be perfectly equal. This approach is done without "snapping" and, therefore, the region won't be aligned with the feature grid:
![[Pasted image 20260428154138.png]]

To do this, we need to sample the features at regularly-spaced points in each sub-region using **bilinear interpolation**.
![[Pasted image 20260428154643.png]]

Since each of the points can't be used directly, we need to compute first a linear combination of the points of the original grid.

## Fast R-CNN vs "Slow" R-CNN
![[Pasted image 20260428154926.png]]

So, Fast R-CNN is very good, but the research for region proposal is still significantly slowing down the performance.

So, there is another approach to address this issue.

## Faster R-CNN: Learnable Region Proposals

Instead of searching the regions manually, we employ a network to find those regions.

![[Pasted image 20260428161107.png]]

So, we insert a **Region Proposal Network (RPN)** to predict the region proposals from the feature map.

### Region Proposal Network (RPN)

Basically, some regions are proposed:
![[Pasted image 20260428162240.png]]
Those proposed regions are **anchor boxes** with a fixed size and they are identified by the position of their center point.

But not all of them are good candidates to contain an object, only few of them are (if not only one).
![[Pasted image 20260428162354.png]]

The task that predicts if a certain proposed region contains an object or not is a binary task and only those regions that have a sufficiently high score will be picked for object detection.

Then, after we've found a region that actually contains an object, we can adjust its size and position to match the object properly. Although, it might result in a region that doesn't perfectly align with the feature grid:
![[Pasted image 20260428162739.png]]

However, there is a major problem, which is illustrated in the following image:
![[Pasted image 20260428162904.png]]

## Faster R-CNN: conclusion

![[Pasted image 20260428163014.png]]

With this approach we have to jointly train with 4 different losses:
1. **RPN classification**: anchor box is object or not an object.
2. **RPN regression**: predict transform from anchor box to proposal box.
3. **Object classification**: classify proposals as background or object class.
4. **Object regression**: predict transform from proposal box to object box.

Speed comparison:
![[Pasted image 20260428163247.png]]

Faster R-CNN is effectively a **two-stage** object detector:
![[Pasted image 20260428171500.png]]

>[!question]
>Do we really need the second stage?
>Can't we perform this task in only one stage?

## Single-Stage Object Detection

With this approach, we can perform object detection in one single stage.

The anchor boxes don't have a fixed size, so the object can be found directly from the anchor box.

![[Pasted image 20260428172204.png]]

>[!note] personal note
>It looks like all the anchor boxes have to be located at the center of the image but I'm not 100% sure about this.

>[!info]
>Some priors can be used to know in advanced which shapes are possible for certain objects (and therefore immediately spot the wrong ones).

# Object detection: conclusion

When we choose which model we want to work with we need to find the most appropriate compromise (which depends on the task) between speed and accuracy:
![[Pasted image 20260428172940.png]]

**Takeaways**:
- Two stage method (Faster R-CNN) get the best accuracy, but are slower.
- Single-stage methods (SSD) are much faster, but don't perform as well.
- Bigger backbones improve performance, but are slower.
- Diminishing returns for slower methods.