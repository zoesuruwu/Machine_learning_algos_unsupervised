## Summary: machine learning algorithms for unsupervised data (K-means and PCA)

### Runable scripts: 
1. main.py: comparison between K-means and PCA models
2. San_diego_tract_k_means.ipynb: detailed illustration for k-means algorithm for spatial data (on Census Tracts for the San Diego (CA) metropolitan area)
   - source: https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html
### Models:
1. K-means: 
    - Summary: This clustering algorithm involves sorting observations into groups without any prior idea about what the groups are, thus a unsupervised technique.
    - Prediction: It is not traditionally used for predictions. If desired to see how the model performs, choose a sample with target variable being categorical, with K classes. We can train a model with K clusters (matched), and make prediction on the cluter assignments. Ways to determine performance can be visualization between predicted class with the its true class, and calculate the success rate.
    - Dataset: continous numerical data.
        1. Seed data: supervised, with numebr of classes in the target variable = 3, hence pre-determined k=3. The main task of this dataset is to see how the model performs in the test dataset between original data and scaled data.
        2. San diego tracts (geograpfic) data: unsupervised, task mainly is to identify optimal number of k clusters
    - How to evaluate clusters: Observations in one group may have consistently high scores on some traits but low scores on others.
    - Performance metrics to choose optimal number of cluters (the higher the score, the better the performance):
        1. The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each data point. The Silhouette Coefficient for data point i is: 
         $s_i = (b_i-a_i) / max(a_i, b_i)$. And the Silhouette Coefficientis coumpted by the average of all $s_i$,
        where 
            - $a_i$ is the  the average distance of the data point $i$ to other data points within the same cluster.
            - $b_i$ is the smallest average distance of the data point $i$ to data points in a different cluster. 
        
            The Silhouette Coefficient for each data point ranges from -1 to 1:
            - A value close to +1 indicates that the data point is well-matched to its own cluster and far from other clusters.
            - A value close to 0 indicates that the data point is on or very close to the decision boundary between two clusters.
            - A value close to -1 indicates that the data point is likely assigned to the wrong cluster.
        2. The Calinski Harabasz Score is the ratio between  **between-cluster dispersion** and **within-cluster dispersion**
        3. Elbow plot: evaluate the decrease in **inertia** as the number of clusters to fit increases, where the drop in **inertia** starts to slow down indicates the point of optimal k value. **inertia** is defined as the sum of squared distances of samples to their closest cluster center.
2. PCA:
     - Summary: This algorithm is used commonly for dimensionality reduction while retaining a significant portion of the variability in the data. It doesn't directly incorporate information from the target variable, thus a unsupervised technique.
     - Prediction: After applying PCA to feature data, the data is transformed into a new coordinate system defined by the principal components, which are the eigenvectors of the covariance matrix calculated from the original features. We then use LogisticRegression model with some modifications to evaluate the model performance. In this project, the following modification is applied:
        - LogisticRegression(multi_class='multinomial', solver='lbfgs')

### Additional tutorial 
- Iterative Initial Centroid Search via Sampling for k-Means Clustering: https://www.kdnuggets.com/2018/09/iterative-initial-centroid-search-sampling-k-means-clustering.html
