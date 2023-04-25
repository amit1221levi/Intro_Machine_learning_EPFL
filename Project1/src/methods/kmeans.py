import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
           
        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """       
        self.K = K
        self.max_iters = max_iters

    def init_centers(self,data):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """       
        return np.random.permutation(data)[:int(self.K)]

    def compute_distance(self,data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        return np.array([np.sqrt(((data-i) ** 2).sum(axis=1)) for i in centers]).T

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        return np.argmin(distances, axis=1)
    
    def compute_centers(self, data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        
        values = np.zeros((K,data.shape[1]))
        for k in range(K):
            values[k,:]= np.mean(data[k==cluster_assignments],axis=0)
        return values
        
    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
            # Initialize the centers
        centers = self.init_centers(data)

        # Loop over the iterations
        for _ in range(max_iter):
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            distances = self.compute_distance(data, centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments, self.K)
            # End of the algorithm if the centers have not moved (hint: use old_centers and look into np.all)
            if np.allclose(centers,old_centers):
                break
        return centers, cluster_assignments
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.
        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        self.D, self.C = training_data, training_labels
        return self.predict(training_data)
    

    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(true_labels[cluster_assignments == i]))
            cluster_center_label[i] = label    
        return cluster_center_label
    

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.
        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        centers, cluster_assignments = self.k_means(self.D, self.max_iters)
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(self.C[cluster_assignments == i]))
            cluster_center_label[i] = label    

        distances = self.compute_distance(test_data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        return cluster_center_label[cluster_assignments]