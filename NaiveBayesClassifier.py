import numpy as np
from collections import defaultdict
from math import exp

class NaiveBayesClassifier:
    def __init__(self, voc, labels):
        # i-th row is a vocabulary length parameter for class i
        self.voc = voc
        self.labels = labels
        self.k = len(labels)
        self.paras = np.array([[0.0]*len(voc)]*self.k)
        self.threshold = 0.001

    def score(self,X,Y):
        '''Average precision.'''
        err = 0
        for i in range(len(Y)):
            x, y = X[i], Y[i]
            if self.predict(x) != y: err+=1
        return 1.0-float(err)/len(Y)

    def predict(self, x):
        '''Return the MAP class (0~k-1).'''
        prob = self.predict_prob(x)
        return max([(i,p) for i,p in enumerate(prob)],key=lambda (a,b):(b,a))[0]

    def predict_prob(self, x):
        '''P(Y|x) = softmax(x).
        
        @return
        For each class k: P(Y=k|x)
        '''
        x = np.array(x) + 1
        prob = np.array([0.0]*self.k)
        for i in range(self.k):
            prob[i] = sum(self.paras[i] * x)
        prob -= max(prob)
        prob = np.exp(prob)
        prob /= sum(prob)
        return prob
    
    def train(self, X, Y, C=0.1, step=0.001, max_iter=2, batch_size=100):
        for i in range(0,len(X),batch_size):
            xx, yy = np.array(X[i:i+batch_size]), np.array(Y[i:i+batch_size])
            self.gradient_ascent(xx, yy, C, step, max_iter)

    def gradient(self, X, Y, C=0, step=1):
        '''W_k := W_k + step * SUM{x*[I{y==k} - P(Y=k|x)] + C*W_k}.
        
        @params:
        X: (n,voc) matrix, training data
        Y: (1,n) labels
        C: penalty hyper-parameter for regularization
        step: step length for gradient ascent

        @return:
        If converged: True
        Else: False
        '''
        grad = np.zeros((self.k, len(self.voc)))
        for i,x in enumerate(X):
            y = Y[i]
            # I{y==k}, (1,k) shape
            I = np.zeros(self.k)
            I[y] += 1
            vec = (I-self.predict_prob(x)).reshape((-1,1))
            grad += vec * x
        # max movement for parameter
        step_length = max(sum((step*grad)**2))
        if step_length<self.threshold: return True
        self.paras += C * self.paras + step * grad
        return False

    def gradient_ascent(self, X, Y, C=0.1, step=0.1, max_iter=20):
        cnt = 0
        while self.gradient(X, Y, C, step):
            cnt += 1
            print "Iter ", cnt, " completed."
            if cnt>=max_iter: break

# load dataset
path = "D:\\Resource\\Courses\\cmu_ml\\My_Python_Sol\\GNB\\data\\data\\"
def getV():
    voc, labels = [], []
    with open(path+"vocabulary.txt") as fr:
        for line in fr.readlines():
            voc.append(line.rstrip("\r\n"))
    with open(path+"newsgrouplabels.txt") as fr:
        for line in fr.readlines():
            labels.append(line.rstrip("\r\n"))
    return voc, labels

def getY():
    Y = []
    with open(path+"train.label") as fr:
        for line in fr.readlines():
            # start from 0
            Y.append(int(line.rstrip("\r\n"))-1)
    return Y

def getData():
    Y = getY()
    voc, labels = getV()
    X = [[0]*len(voc)]*len(Y)
    with open(path+"train.data") as fr:
        for line in fr.readlines():
            docId,wordId,cnt = [int(e) for e in line.rstrip("\r\n").split(" ")]
            # start from 0
            docId -= 1; wordId -= 1
            X[docId][wordId] = cnt
    return X[:200], Y[:200]

# unit test
X,Y = getData()
voc, labels = getV()
print "dataset loaded."
c = NaiveBayesClassifier(voc, labels)
c.train(X,Y)
print c.score(X,Y)












