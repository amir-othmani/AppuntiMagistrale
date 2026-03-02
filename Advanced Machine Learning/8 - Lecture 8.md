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
- This will also help the model to generalize to new tasks in the future as a hypothesis space that performs well for a sufficiently large number of training tasks and will also perform well for learning novel tasks
- Acts as a regularization
- No more one-directional transfer but a symmetric two-directional transfer.

## Detecting a single object

![[Pasted image 20251025155055.png]]
## Detecting multiple objects

![[Pasted image 20251025155142.png]]

## Object Detection as Classification (Sliding Windows)

(page 17)