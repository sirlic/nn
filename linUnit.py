# __* coding:utf-8 *__

import numpy as np

class LinearUnit(object):

    def __init__(self,input_num,loop,step,minError):
        '''
        初始化线性单元，
        '''
        self.weights = np.zeros(input_num)
        self.loop = loop
        self.step = step
        self.minError = minError
        self.bias = 0

    def train(self,input_vecs,labels):
        for i in range(0,self.loop):
            loss = self.__loss(input_vecs,labels)
            gradient = np.dot(loss,input_vecs)
            self.weights = self.weights - self.step*gradient
            self.bias = self.bias - self.step*gradient

    def __loss(self,input_vecs,labels):
        print self.weights
        print np.dot(input_vecs,self.weights)
        return labels - (np.dot(input_vecs,self.weights)+self.bias)

    def __str__(self):
        print self.weights
        print self.bias
if __name__ == '__main__':
    linear = LinearUnit(1,1000,0.1,0)
    input_vecs = np.arange(0,100,1).reshape((100,1))
    labels = 2*np.arange(0,100,1)+np.random.f(1,2)
    linear.train(input_vecs,labels)
    print linear
