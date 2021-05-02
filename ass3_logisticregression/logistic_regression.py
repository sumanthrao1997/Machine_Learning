import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from core import *
import typing

def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training and validation sets

    Args:
        - x (np.ndarray): set of points x1 and x2
        - y (np.ndarray): class labels of each sample

    Returns:
        - X_train, Y_train (np.ndarray, np.ndarray): input data and labels to be used for training
        - X_val, Y_val (np.ndarray, np.ndarray): input data and labels to be used for validation
    """
    


    percent = int(0.8*np.shape(x)[0]) #splitting 10%  data to validation set
    rand = random.choice(np.shape(x)[0], size = percent ,replace = False)    #generating random numbers 
    np.sort(rand)
    x_train = x[rand,:]
    y_train = y[rand,:]
    x_val = np.delete(x,rand , axis=0)
    y_val = np.delete(y,rand, axis=0)

    return x_train, y_train, x_val, y_val

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    This function should calculate the sigmoid activation function w.r.t. x

    Args:
        - x (np.ndarray): vector with float values to calculate the sigmoid function

    Returns:
        - sigmoid_x (np.ndarray): output vector with the values sigmoid(x)
    """
    sigmoid_x = 1/(1+np.exp(-x))

    return sigmoid_x

def softmax(x: np.ndarray) -> np.ndarray:
    """
    This function should calculate the softmax activation function w.r.t. x

    Args:
        - x (np.ndarray): vector with float values to calculate the softmax function

    Returns:
        - softmax_x (np.ndarray): output vector with the values softmax(x)
    """

    x -= np.max(x)
    softmax_x = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T

    return softmax_x

class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = [0,0,0]

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
             
        y_pred = self.theta[0] + self.theta[1]*x[:,0] + self.theta[2]*x[:,1]
        
        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der (np.ndarray): first derivative value
        """
        
        

        der = [0,0,0]
        sig = sigmoid(y_pred)
        for i in range(np.shape(x)[0]):
            if(y[i]==1):
                    der = der + (sig[i] - 1)*np.append([1],x[i])
            else:
                    der = der + sig[i]*np.append([1],x[i])
        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta parameters that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """

        for i in range(self.epochs):
            y_pred = self.predict_y(x)
            self.theta = self.theta - self.learning_rate *self.first_derivative(x,y_pred,y)

            i = i+1


    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """
        y_pred = self.predict_y(x)
        s = sigmoid(y_pred)
        acc = accuracy(y,s.reshape(x.shape[0],1))

        return acc

class MultiClassLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_class = np.zeros((3,3))

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """

        y_pred = x @ self.theta_class.T
        
        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y,
        for each possible class.

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der: first derivative value
        """
        der = np.zeros((3,3))
        softmx = softmax(y_pred)
        
        ones = np.ones((np.shape(x)[0],1))
        x_append = np.hstack((ones , x))
        
        y1 = softmx - y 
        der = y1.T @ x_append

        return der


    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta_class parameters (multiclass) that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """
        ones = np.ones((np.shape(x)[0],1))
        x_append = np.hstack((ones , x))
        for i in range(self.epochs):
            y_pred = self.predict_y(x_append)
            self.theta_class = self.theta_class - self.learning_rate * self.first_derivative(x,y_pred,y)
            i = i+1



    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred)) note: accuracy function already implemented in core.py
        """
        ones = np.ones((np.shape(x)[0],1))
        x_append = np.hstack((ones , x))
        y_pred = self.predict_y(x_append)
        s = softmax(y_pred)
        acc = accuracy(y,s)

        return acc