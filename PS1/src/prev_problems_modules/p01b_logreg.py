import numpy as np
import util
from linear_model import LinearModel

class LogisticRegression(LinearModel):

    def fit(self, x, y):
        # we need to set theta size, based on x 
        # n is the number of params 
        # m is the number of training examples 
        m, n = x.shape
        self.theta = np.zeros((1, n))


        for i in range(self.max_iter):
            prev_theta = self.theta 
            self.theta = prev_theta - np.dot(np.linalg.inv(self.hessian(x, y)), self.gradient(x, y).T).T
            # check if the val is less than eps 
            if np.linalg.norm(self.theta - prev_theta, ord=1) < self.eps:
                break
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # need gradient 
    def H(self, x):
        # definition of forward 
        return self.sigmoid(np.dot(self.theta, x.T))
    
    def gradient(self, x, y):
        m, _ = x.shape
        return (np.dot(self.H(x) - y, x)) / m
    
    def hessian(self, x, y):
        # goal is a 3 by 3 matrix 
        m, _ = x.shape        
        return (x.T * self.H(x) * (1 - self.H(x))) @ x / m

    def predict(self, x):
        return self.sigmoid(np.dot(self.theta, x.T))