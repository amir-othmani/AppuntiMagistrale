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
(continue from page 66)