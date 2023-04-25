import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        
        self.training_data, self.training_labels = training_data, training_labels
        
        D = training_data.shape[1]  # number of features
        C = label_to_onehot(training_labels).shape[1]  # number of classes
        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))
        for _ in range(self.max_iters):
            
            gradient = self.gradient_logistic_multi(training_data, label_to_onehot(training_labels), self.weights)
            self.weights = self.weights - self.lr*gradient
    
            data_point_probability = self.f_softmax(training_data, self.weights)
        
            predictions = np.argmax(data_point_probability, axis = 1) 
            
            if self.accuracy_fn(predictions, training_labels) == 1:
                break
         
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        
        data_point_probability = self.f_softmax(test_data, self.weights)
        
        pred_labels = np.argmax(data_point_probability, axis = 1)                                 
                                           
        return pred_labels
    
    
    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic 
        regression.
    
        Args:
            data (array): Input data of shape (N, D)
        
            W (array): Weights of shape (D, C) where C is the
            number of classes
        Returns:
            array of shape (N, C): Probability array where 
            each value is in the range [0, 1] and each 
            row sums to 1.
            The row i corresponds to the prediction of the ith 
            data sample, and 
            the column j to the jth class. So element [i, j] 
            is P(y_i=k | x_i, W)
        """

        x_w = np.exp(data @ W)
        softmax = (x_w/(np.sum(x_w, axis=1, keepdims = True)))
                                                                                                                
        return softmax    
    

    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
    
        softmax_predictions = self.f_softmax(data, W)
    
        gradient_logistic_multi = data.T @ (softmax_predictions - labels)
        
        return gradient_logistic_multi
    
    def accuracy_fn(self, labels_pred, labels_gt):
        """
        Computes the accuracy of the predictions (in percent).
        
        Args:
            labels_pred (array): Predicted labels of shape (N,)
            labels_gt (array): GT labels of shape (N,)
        Returns:
            acc (float): Accuracy, in range [0, 1].
        """
        return np.mean(labels_pred == labels_gt)*100