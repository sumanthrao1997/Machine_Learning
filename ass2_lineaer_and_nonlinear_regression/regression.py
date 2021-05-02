import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from typing import Optional

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """
    err = np.sqrt(np.sum(np.square((y - y_pred)))/len(y))

    return err

def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training, validation

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    percent = int(0.1*len(x)) #splitting 10%  data to validation set
    rand = random.randint(len(x)-1, size = percent)    #generating random numbers 
    np.sort(rand)
    x_train = x
    y_train = y
    x_val = []
    y_val = []
    x_train = np.delete(x_train,rand)
    y_train = np.delete(y_train,rand)
    for i in rand:
        x_val = np.append(x_val, x[i]) 
        y_val = np.append(y_val, y[i])
    return x_train ,y_train,x_val , y_val


class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line

        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        #Design matrix
        one =  np.ones(len(x)).reshape(len(x),1)
        x_design = np.hstack((x.reshape(len(x),1) , one ))

        #Normal equations
        N= np.matmul(x_design.T ,x_design)
        n = np.matmul(x_design.T , y.reshape(len(y),1))

        #parameters theta0 and theata1
        parameter = np.matmul(np.linalg.inv(N) , n)

        self.theta_0 = parameter[0]
        self.theta_1 = parameter[1]
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """
        #predicting y
        y_pred = []
        for i in x:
            y_pred = np.append(y_pred, self.theta_0*i + self.theta_1 )

        return y_pred

class NonLinearRegression:
    def __init__(self):
        self.theta = None
        self.degree = None
    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
        #design matrix
        x_design =np.empty(degree+1)
        for i in x:
            temp = [i**j for j in range(degree+1)]  #generating the row
            x_design = np.vstack((x_design,[temp]))  #building design matrix
        
        x_design = x_design[1: ,:] #final design matrix

        #normal equations
        N= np.matmul(x_design.T ,x_design)
        n = np.matmul(x_design.T , y.reshape(len(y),1))
        
        #parameters theta matrix
        self.theta = np.matmul(np.linalg.inv(N) , n)
        self.degree = degree
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """
        y_pred = []
        for i in x:
            temp = [self.theta[j]* (i**j) for j in range(self.degree+1)]
            y_pred = np.append(y_pred,np.sum(temp) )


        return y_pred
