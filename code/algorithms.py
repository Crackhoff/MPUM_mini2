import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=25000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def _predict_prob(self, X): # X is a vector of features
        z = np.dot(X, self.theta)
        return self.sigmoid(z)
    
    def _predict(self, X):
        return self._predict_prob(X) > 0.35
              
    def fit(self, X, y, reset=True):
        # gradient descent algorithm for thetas
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros(X.shape[1]) # looks like it behaves better than random
        # self.theta = np.random.rand(X.shape[1])
        
        if reset:
            self.history = []
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient
            
            self.history.append(self.loss(h, y))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def accuracy(self, y_hat, y):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(y.shape[0]):
            # get ith row of X
            # print(pred, y_pred[i])
            if y[i] == y_hat[i]:                        
                true_positive += 1
            elif y[i] and not y_hat[i]:
                false_negative +=1
            elif y_hat[i] and not y[i]:
                false_positive +=1
            else:
                true_negative +=1
        return (true_positive + true_negative) / y.shape[0] * 100, true_positive/(true_positive + false_positive) * 100, true_positive/(true_positive + false_negative) * 100
     
    def predict(self,X):
        X = np.insert(X, 0, 1, axis=1)
        res = []
        for i in range(X.shape[0]):
            res.append(self._predict(X[i]))
        return res            
    
class NaiveBayes:
    '''
    This class implements the Naive Bayes algorithm for classification.
    it assumes, that the features are independent given the class.
    the model predicts binary (0,1) outcome out of features of 10 classes each.
    '''

    def _predict_prob(self, X):
        # calculate probability of y = 1
        # X is a vector of features
        
        # calculate likelihoods
        likelihood_1 = 1
        likelihood_0 = 1
        log_likelihood_1 = 0
        log_likelihood_0 = 0
        for j in range(X.shape[0]):
            likelihood_1 *= self.likelihoods[j][(1, X[j])]
            likelihood_0 *= self.likelihoods[j][(0, X[j])]
            log_likelihood_1 += self.log_likelihoods[j][(1, X[j])]
            log_likelihood_0 += self.log_likelihoods[j][(0, X[j])]
        
        log_likelihood_1 += np.log(self.prior)
        log_likelihood_0 += np.log(1 - self.prior)
        
        log_prob_1 = log_likelihood_1 - np.log(np.exp(log_likelihood_1) + np.exp(log_likelihood_0))
        prob_log = np.exp(log_prob_1)
    
        prob_1 = likelihood_1 * self.prior
        prob_0 = likelihood_0 * (1 - self.prior)
        prob_normal = prob_1 / (prob_1 + prob_0)
        # print(prob_log, prob_normal, abs(prob_normal-prob_log)) # it seems that those values are really similar ( hardly ever there is decisive difference)
        # return prob_normal
        return prob_log

    def _predict(self, X):
        return self._predict_prob(X) >= 0.40 # is this legal? I don't know, but it works :)
    
    def fit(self, X, y, reset=True):
        # calculate prior probabilities
        self.prior = (np.sum(y) + 1) / (len(y) + 2)

        X_Y = np.column_stack((X, y))
        
        X_0 = X_Y[X_Y[:, -1] == 0][:, :-1]
        
        X_1 = X_Y[X_Y[:, -1] == 1][:, :-1]
            
        if reset:    
            self.likelihoods = []     
            self.log_likelihoods = []  
        # calculate likelihoods
        for i in range(X.shape[1]): # for each feature
            likelihood = {}
            log_likelihood = {}
            for j in range(1,11): # for each class
                count_1 = np.sum(X_1[:, i] == j)
                count_0 = np.sum(X_0[:, i] == j)
                likelihood[(1,j)] = (count_1 + 1) / (len(X_1) + 9)
                likelihood[(0,j)] = (count_0 + 1) / (len(X_0) + 9)
                log_likelihood[(1,j)] = np.log(likelihood[(1,j)])
                log_likelihood[(0,j)] = np.log(likelihood[(0,j)])
            self.likelihoods.append(likelihood)
            self.log_likelihoods.append(log_likelihood)
        
    def accuracy(self, y_hat, y):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i in range(y.shape[0]):
            if y[i] and y_hat[i]:
                true_positive += 1
            elif y[i] and not y_hat[i]:
                false_negative +=1
            elif y_hat[i] and not y[i]:
                false_positive +=1
            else:
                true_negative +=1
        return (true_positive + true_negative) / y.shape[0] * 100, true_positive/(true_positive + false_positive) * 100, true_positive/(true_positive + false_negative) * 100
    
    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            res.append(self._predict(X[i]))        
        return res
    
    