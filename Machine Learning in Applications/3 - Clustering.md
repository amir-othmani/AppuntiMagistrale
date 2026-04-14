# Proximity measures
**Proximity** is a generic term that refers to either similarity or dissimilarity.

**Similarity**:
- Numerical measure of how alike two data objects are.
- Measure is higher when objects are more alike.
- Often falls in range [0,1].

**Dissimilarity**:
- Numerical measure of how different two data objects are.
- Measure is lower when objects are more alike.
- Minimum dissimilarity often 0, upper limit varies.
- **Distance** sometimes used as a synonym, usually for specific classes of dissimilarities.

# Clustering strategies
- Partitional (flat) clustering
	- Iteratively re-assign points to a finite set of disjoint clusters.
	- Example: k-means (and variants).
- Hierarchical clustering
	- Iteratively merge or split a set of nested clusters, organized into a hierarchical tree.
- Density based clustering
	- Partitions data based on density.
	- Examples: Mean-shift, DBSCAN.

# Partitional clustering
## K-means algorithm
1. For each cluster, compute the centroid with an arithmetic mean.
2. Then, for each centroid, find the closest point with the euclidean distance (observation).

Iterate above two steps until convergence.
Works towards the minimization of the within-cluster scatter (total sum of point-to-centroid distances).
Also called **SSE**: sum of the squared distance.

Example:
![[fig_3-1.png]]


Disadvantages:
- Dependent of initialization
	- i.e. you need to choose in advance how many centroids you have
- Needs many iterations
- Sensitive to outliers
	- big issue, since outliers are common
	- can use k-medians to address this issue
- How to decide K?

### Deciding K
The way to decide is to make a plot with different K values and see what happens:
![[fig_3-2.png]]

The optimal number is the one that has the most dramatic change w.r.t. the previous value and is reasonably low enough.
In this example, the optimal K value is 2.

>[!warning]
>Problem of K-means: it needs an already good distribution of the data, where the elements are somewhat clearly separate.
>If there is no clear separation, K-means is probably going to fail.

## Hierarchical clustering
Produces a set of nested clusters organized as a hierarchical tree.
Can be visualized as a dendrogram:
- A tree like diagram that records the sequence of merges or splits.

![[fig_3-3.png]]

Strength of hierarchical clustering:
- Do not have to assume any particular number of clusters
	- Any desired number of clusters can be obtained by ‘cutting’ the dendogram at the proper level.
- They may correspond to meaningful taxonomies
	- Example in biological sciences (e.g., animal kingdom, phylogeny reconstruction, …).

Although, it can be unclear how many clusters there are, because the notion of clustering itself is ambiguous:
![[fig_3-4.png]]

There are two types of hierarchical clustering:
- Agglomerative:
	- Start with the points as individual clusters.
	- At each step, merge the closest pair of clusters until only one cluster (or k clusters) left.
- Divisive:
	- Start with one, all-inclusive cluster.
	- At each step, split a cluster until each cluster contains a point (or there are k clusters).

>[!todo]
>From page 27 to page 62 read the slides.

Problems and limitations:
- Once a decision is made to combine two clusters, it cannot be undone.
- No objective function is directly minimized.
- Different schemes have problems with one or more of the following:
	- Sensitivity to noise and outliers
	- Difficulty handling different sized clusters
	- Breaking large clusters
- Inherently unstable towards addition or deletion of samples.

To get the desired number of clusters, we just need to choose the max threshold of distance:
![[fig_3-5.png]]

# Density-based clustering
## Mean shift clustering
It is based on the concept of **kernel density estimation (KDE)**.
The data points (i.e. the feature vectors of the data to be clusterized) are interpreted as they were sampled from a probability distribution. KDE is a
method to estimate the underlying probability distribution.

## Kernel density estimation
It works by placing a **kernel** (i.e. a weighting function) on each point of the dataset in the feature space. Adding all of the individual kernels up generates a probability surface (e.g., density function).
$$ \hat f_h(x) = \frac1{nh} \sum\limits_{i=1}^n K(\frac{x-x_i}h) $$

The most popular kernel function is the Gaussian kernel:
$$ K(\frac{x-x_i}h)=\frac1{\sqrt{2\pi}}e^{-\large\frac{(x-x_i)^2}{2h^2}} $$

## From KDE to clustering
Mean shift exploits this KDE idea by imagining what the points would do if they all climbed up hill to the nearest peak on the KDE surface.
It does so by iteratively shifting each point uphill until it reaches a peak.
Hence, all the points are clustered depending on the peak they are shifted to.

## Mean shift clustering
The mean shift algorithm seeks modes of the given set of points:
1. Choose a kernel and a bandwidth value
2. For each point:
	- Center a (density estimator) window on that point
	- Compute the mean shift vector m of the data in the search window
	- Move the density estimation window by m
	- Repeat (b,c) until convergence
3. Assign points that lead to the same mode to the same cluster.

Pros:
- There is no need to apriori decide the number of clusters
- Robust to outliers
- Quite flexible, no assumptions on the data distribution
Cons:
- Computationally intensive
- Not feasible for high-dimensional dataset
- Results depend on kernel/bandwith

## DBSCAN
DBSCAN is a density-based algorithm.
- Density = number of points within a specified radius (Eps)
- A point is a **core point** if it has more than a specified number of points (MinPts) within Eps.
	- These points are in the interior of a cluster.
- A **border point** has fewer than MinPts within Eps, but is in the neighborhood of a core point.
- A **noise point** is any point that is not a core point or a border point.

![[fig_3-6.png]]

The algorithm works in this way:
1) Label all points as core, border, or noise points.
2) Eliminate noise points.
3) Put an edge between all core points that are within Eps of each other.
4) Make each group of connected core points into a separate cluster.
5) Assign each border point to one of the clusters of its associated core points.

Pros:
- DBSCAN is resistant to noise.
- It can handle clusters of different shapes and sizes.
Cons:
- DBSCAN doesn't work well when there are varying densities and/or high-dimensional data.

# Assessing clustering validity
Since there is no supervision there is no universal way to tell the quality of the clustering.
So, there are many different types of clustering validation:
1. Determining the clustering tendency of a set of data, i.e., distinguishing whether non-random structure actually exists in the data.
2. Comparing the results of a cluster analysis to externally known results, e.g., to externally given class labels.
3. Evaluating how well the results of a cluster analysis fit the data without reference to external information.
	- Use only the data
4. Comparing the results of two different sets of cluster analyses to determine which is better.
5. Determining the ‘correct’ number of clusters.

For 2, 3, and 4, we can further distinguish whether we want to evaluate the entire clustering or just individual clusters.

Numerical measures used to judge various aspects of cluster validity are classified into the following three types:
- **External index**: Measures extent to which cluster labels match externally supplied class labels.
- **Relative index**: Compares two different clusterings or clusters.
	- Often an external or internal index is used for this purpose.
- **Internal index**: Measures the “goodness” of a clustering structure without respect to external information.
	- Correlation
	- Visualize similarity matrix
	- Sum of Squared Error (SSE)
	- Cohesion and Separation
	- Silhouette

>[!todo]
>From page 93 to the end just read the slides.

