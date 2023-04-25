import argparse

import numpy as np 
import torch
import time
from torch.utils.data import DataLoader

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn

import random


import itertools








#===============================Random_search=====================================================================================================================
def random_search_logistic_regression(xtrain, ytrain, xtest, ytest, n_iter=20):
    learning_rates = [10**(-i) for i in range(6)]
    max_iters = [100*i for i in range(1, 11)]

    best_acc = 0
    best_lr = 0
    best_max_iter = 0

    for _ in range(n_iter):
        lr = random.choice(learning_rates)
        max_iter = random.choice(max_iters)

        method_obj = LogisticRegression(lr, max_iter)
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)

        acc = accuracy_fn(preds, ytest)

        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_max_iter = max_iter

    return best_lr, best_max_iter
#========================================================================================================================================================================================================





def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """

    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data('/Users/amitlevi/Desktop/Machine learning git/sciper1_sciper2_sciper3_project/src/dataset_HASYv2')
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)
    #if args.method!="svm":
    xtrain = normalize_fn(xtrain,xtrain.mean(1, keepdims=True),xtrain.std(1, keepdims=True))
    xtest = normalize_fn(xtest,xtest.mean(1,keepdims=True),xtest.std(1,keepdims=True))


    
    # Make a validati
    # on set (it can overwrite xtest, ytest)
    if not args.test:
        print(xtest.shape[0])
        index = np.random.permutation(xtest.shape[0])[:args.nb]
        xtest = np.take(xtrain,index,axis=0)
        ytest = np.take(ytrain,index)

        xtrain = np.delete(xtrain,index,axis=0)
        ytrain = np.delete(ytrain,index)
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)
    elif args.method == "kmeans":
        
        trainingTabacc = np.array([])
        validationTabacc = np.array([])
        trainingTabF1 = np.array([])
        validationTabF1 = np.array([])
        if args.findBestK :
            for i in range(1,args.K):
                method_obj = KMeans(i, args.max_iters) 
                preds_train = method_obj.fit(xtrain, ytrain)
                    
                # Predict on unseen data
                preds = method_obj.predict(xtest)


                ## Report results: performance on train and valid/test sets
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                print("For K = %d", i)
                print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                trainingTabacc= np.append(trainingTabacc,acc)
                trainingTabF1 = np.append(trainingTabF1,macrof1)
                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                validationTabacc= np.append(validationTabacc,acc)
                validationTabF1 = np.append(validationTabF1,macrof1)
                print("---------------------------------------------------")
        elif args.findBestmean:
            a = np.array([])
            b = np.array([])
            for _ in range(20):
                method_obj = KMeans(args.K, args.max_iters) 
                preds_train = method_obj.fit(xtrain, ytrain)
                    
                # Predict on unseen data
                preds = method_obj.predict(xtest)

                
                ## Report results: performance on train and valid/test sets
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                a=np.append(a,macrof1)
                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                print("---------------------------------------------------")
                b=np.append(b,macrof1)

            print(a)
            print(b)    
        else:    
            method_obj = KMeans(args.K, args.max_iters)
        print("TrainingAcc", trainingTabacc)   
        print("TrainingF1", trainingTabF1)   
        print("ValidationAcc", validationTabacc)   
        print("VallidationF1", validationTabF1)   
         
    elif args.method == "logistic_regression":

        trainingTabAcc = np.array([])
        validationTabAcc = np.array([])
        trainingTabF1 = np.array([])
        validationTabF1 = np.array([])
        timeCost = np.array([])
        
   
        
        
        if args.findBestLr:
            for i in (10**p for p in range(-5, 1)):
                start_time = time.time()
                method_obj = LogisticRegression(i, args.max_iters)
                preds_train = method_obj.fit(xtrain, ytrain)

                # Predict on unseen data
                preds = method_obj.predict(xtest)

                ## Report results: performance on train and valid/test sets
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                print("For lr = %d", i)
                print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                trainingTabAcc = np.append(trainingTabAcc, acc)
                trainingTabF1 = np.append(trainingTabF1, macrof1)
                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                validationTabAcc = np.append(validationTabAcc, acc)
                validationTabF1 = np.append(validationTabF1, macrof1)
                timeCost = np.append(timeCost, time.time() - start_time)
                print(f"Elapsed time: {time.time() - start_time:.2f} seconds")  # compute elapsed time
                print("---------------------------------------------------")
        elif args.findBestMaxIter:
            
            trainingTabAcc = np.array([])
            validationTabAcc = np.array([])
            trainingTabF1 = np.array([])
            validationTabF1 = np.array([])

            '''
            for i in range(1,10):
                start_times = time.time()
                max_iters = 100*i
                method_obj = LogisticRegression(args.lr, max_iters)
                preds_train = method_obj.fit(xtrain, ytrain)

                # Predict on unseen data
                preds = method_obj.predict(xtest)
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                print("For max iter = %d", max_iters)
                print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                trainingTabAcc = np.append(trainingTabAcc, acc)
                trainingTabF1 = np.append(trainingTabF1, macrof1)
                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                validationTabAcc = np.append(validationTabAcc, acc)
                validationTabF1 = np.append(validationTabF1, macrof1)
                timeCost = np.append(timeCost, time.time() - start_times)
                print(f"Elapsed time: {time.time() - start_times:.2f} seconds")  # compute elapsed time
                print("---------------------------------------------------")
            '''
            for i in range(0,50):
                method_obj = LogisticRegression(args.lr)
                preds_train = method_obj.fit(xtrain, ytrain)
                preds = method_obj.predict(xtest)
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                trainingTabAcc = np.append(trainingTabAcc, acc)
                trainingTabF1 = np.append(trainingTabF1, macrof1)
                acc = accuracy_fn(preds, ytest)
                macrof1 = macrof1_fn(preds, ytest)
                validationTabAcc = np.append(validationTabAcc, acc)
                validationTabF1 = np.append(validationTabF1, macrof1)
            print("TACC mean", np.mean(trainingTabAcc))
            print("VAD mean", np.mean(validationTabAcc))
            print("TACC f1 mean", np.mean(trainingTabF1))
            print("VAD f1 mean", np.mean(validationTabF1))
            

        else:
            method_obj = LogisticRegression(args.lr, args.max_iters)
        print("TrainingAcc", trainingTabAcc)
        print("TrainingF1", trainingTabF1)
        print("ValidationAcc", validationTabAcc)
        print("VallidationF1", validationTabF1)
        print("time cost", timeCost)

    elif args.method == "svm":
        if args.grid_search:
                from sklearn.model_selection import GridSearchCV
                from sklearn import svm
                import matplotlib.pyplot as plt




                # Define the range of hyperparameters to search
                param_grid = {
                'C': [0.1, 90, 150],
                'kernel': ['rbf','poly'],
                'degree': [ 3, 4],
                'gamma': ['scale', 'auto', 30, 0.02],
                'coef0': [0, 1, 2]
   
   
                }

                # Create an SVM classifier with the default parameters
                svm_clf = svm.SVC()

                # Create the GridSearchCV object
                grid_search = GridSearchCV(svm_clf, param_grid, scoring='accuracy', cv=5)


                # Fit the data to the GridSearchCV object
                grid_search.fit(xtrain, ytrain)

                # Get the best parameters found by the grid search
                best_params = grid_search.best_params_

                # Update the SVM classifier with the best parameters
                svm_clf.set_params(**best_params)



                # Get the results of the grid search
                results = grid_search.cv_results_
                # Plot the results
                plt.plot(results['mean_test_score'])
                plt.xlabel('Hyperparameter Combination')
                plt.ylabel('Accuracy')
                plt.show()
                # Print the best parameters
                print("Best parameters found by grid search:", best_params)
        else:
                svm_clf = svm.SVC(C=args.svm_c, kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0)



        if args.findBestArgs:
            tabC = [0.001,0.01,0.1,1,10,100,150,200,500,1000]
            tabG = [0.001,0.01,0.1,1,10]
            tabK = ['linear','poly','rbf']
            for k in tabK:
                for i in tabC:
                    for j in tabG:
                        method_obj = SVM(C = i, gamma = j,kernel = k, degree= args.svm_degree,coef0= args.svm_coef0)
                        
                        preds_train = method_obj.fit(xtrain, ytrain)
                            
                        # Predict on unseen data
                        preds = method_obj.predict(xtest)


                        ## Report results: performance on train and valid/test sets
                        acc = accuracy_fn(preds_train, ytrain)
                        macrof1 = macrof1_fn(preds_train, ytrain)
                        print(f"\n - C = {i:.6f}- Gamma = {j:.6f}- Kernel = " + k)
                        print(f"Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

                        acc = accuracy_fn(preds, ytest)
                        macrof1 = macrof1_fn(preds, ytest)
                        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                        print("---------------------------------------------")
        else:
            method_obj = SVM(args.svm_c, args.svm_kernel,args.svm_gamma, args.svm_degree, args.svm_coef0)
    

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
        
    # Predict on unseen data
    preds = method_obj.predict(xtest)


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    # Feel free to add more arguments here if you need!

    parser.add_argument("--nb", default=200, type=int, help="number of element to put in the validation set" )
    parser.add_argument("--findBestK", action="store_true", help="if we have to do the iterations to find the best K" )
    parser.add_argument("--findBestmean", action="store_true", help="calculate the means" )

    parser.add_argument("--grid_search", action="store_true", help="calculate the means" )

    parser.add_argument("--findBestLr", action="store_true", help="calculate the learning rate")
    parser.add_argument("--findBestMaxIter", action="store_true", help="calculate the best number of iteration")
    parser.add_argument("--findBestArgs", action="store_true", help="calculate arguments for SVM")

    
    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
