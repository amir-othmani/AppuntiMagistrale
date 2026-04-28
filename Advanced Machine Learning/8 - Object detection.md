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

(CONTINUE FROM HERE)
## Fast R-CNN

It's a variant of R-CNN designed to be faster than regular R-CNN. The idea is to run CNN before warping.

>[!todo]
>What is warping???
>I'm pretty sure this word was never used before in the slides.

The approach is shown in this picture:
![[Pasted image 20260306150042.png]]

### Cropping features: RoI Pool

>[!todo]
>I'm gonna read the slides for this part, too hard to take notes on this (because it would require too many screenshots).
>
>This part goes from page 70 to the end (I'm fed up and I wanna go to the next chapter).



