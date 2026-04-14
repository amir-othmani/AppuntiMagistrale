# No free lunch theorem
There is no model that works best for every problem.
So, the model needs to be chosen depending on the context.
# Choosing a classifier
When we choose a classifier, we need to take into account three parameters:
- **Training error**
- **Test error**
- **Generalization**

When we evaluate the quality of a model, there are many factors to consider:
- **Bias**: tells us how the estimated model differs from the true model of the real-world, which is mainly due to inaccurate assumptions or necessary simplifications.
- **Variance**: tells us how much varies the output when we change the training set in the model.
- **Underfitting**: happens when the model is too "simple" and so it's not able to represent the real-world correctly, when this happens the model has:
	- High training error and high test error.
	- High bias and low variance.
- **Overfitting**: happens when the model is too "complex" and matches almost perfectly the training data; this means the model took into account irrelevant characteristics that should've been considered as noise. In this case the model has:
	- Low training error and high test error.
	- Low bias and high variance.

(page 8)