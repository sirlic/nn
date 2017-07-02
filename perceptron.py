# __* coding:utf-8 *__
import numpy as np
import matplotlib.pyplot as plt

'''
感知器
'''
class Perception(object):
    def __init__(self,input_num,activator):
        '''
        初始化感知器，设置输入参数，以及激活函数。
        激活函数的类型为double -> double
        activator 激活函数
        '''
        self.activator = activator
        #权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        #偏置项初始化为0
        self.bias = 0.0

       # 画图用 
       # self.x = np.arange(-1,10,0.01)
    def __str__(self):
        '''
        打印学习到的权重，偏置项
        '''
        return 'weight \t: %s \nbias \t:%f\n' % (self.weights,self.bias)
    def predict(self,input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把 input_vec[x1,x2,x3...] 和 weights[w1,w2,w3...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1,x2*w2,x3*w3,...]
        # 最后利用reduce求和
        return self.activator(
                reduce(lambda a,b: a+b,
                    map(lambda (x,w):x*w,
                        zip(input_vec,self.weights))
                    ,0.0) + self.bias)

    def train(self,input_vecs,lables,iteration,rate):
        '''
        输入训练数据：一组向量，与每个向量对应的label；以及训练轮数，学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs,lables,rate)

    def _one_iteration(self,input_vecs,lables,rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec,label),...]
        # 把每个训练样本是(input_vec,labels)
        samples = zip(input_vecs,lables)

        # 对每个样本，按照感知器规则更新权重
        for (input_vec,lable) in samples:
            #计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec,output,lable,rate)
    
    def _update_weights(self,input_vec,output,label,rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...] 和weights[w1,w2,w3,...] 打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = map(
                lambda (x,w): w + rate*delta*x,
                zip(input_vec,self.weights))
        # 更新bias
        self.bias += rate*delta
      #  y = self.weights*self.x+self.bias
       # plt.plot(self.x,y)
        #plt.pause(0.01)

def f(x):
    '''
    激活函数
    '''
    return  1 if x > 0 else 0
    #return x

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输出向量列表
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    lables = [1,0,0,0]
    return input_vecs,lables

def get_one_data():
    input_vecs = [[1],[3],[5],[7],[9]]
    x = [1,3,5,7,9]
    lables = [3,7,11,15,19]
    plt.scatter(x,lables,s=25)
    return input_vecs,lables

def train_and_perception():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2 （因为and是二元函数），激活函数为f
    p = Perception(22,f)
    # 训练， 迭代10轮，学习速率为0.1
    input_vecs,lables = get_training_dataset()
    #input_vecs,lables = get_one_data()
    p.train(input_vecs,lables,10,0.1)
    #返回训练好的感知器
    return p


if __name__ == '__main__':
    #  训练and感知器
    and_preception = train_and_perception()
    # 打印训练获得的权重
    print and_preception
    #plt.show()
    # 测试
    
