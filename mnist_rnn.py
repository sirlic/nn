
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data',one_hot=True)


# In[3]:


model_path = "./model_mnist_rnn.ckpt"
#一张图片是28*28，run把它分块
chunk_size = 28
chunk_n = 28
rnn_size = 256
n_output_layer =10

X = tf.placeholder('float',[None,chunk_n,chunk_size])
Y = tf.placeholder('float')


# In[4]:


# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size,n_output_layer])),
                                               'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
    data = tf.unstack(data,axis=1)
    outputs,status = tf.nn.static_rnn(lstm_cell,data,dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1],layer['w_']),layer['b_'],name='output')
    return output


# In[5]:


batch_size = 100

def train_neural_network(x,y):
    predict = recurrent_neural_network(x)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost_func)
    saver = tf.train.Saver()  
    epochs = 13
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples/batch_size)):
                x,y = mnist.train.next_batch(batch_size)
                x = x.reshape(-1,chunk_n,chunk_size)
                _,c = sess.run([optimizer,cost_func],feed_dict={X:x,Y:y})
                epoch_loss += c
            print(epoch," : ",epoch_loss)
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        saver.save(sess,model_path)
        print('准确率: ', accuracy.eval({X:mnist.test.images.reshape(-1, chunk_n, chunk_size), Y:mnist.test.labels}))


# In[6]:


train_neural_network(X,Y)


# In[28]:


import numpy as np
get_ipython().magic(u'matplotlib inline')

import io

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
saver = tf.train.Saver()
with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    saver.restore(sess,'./mnist_rnn.ckpt')
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("output:0")
    y = sess.run(output,feed_dict={X:mnist.test.images[0:8].reshape(-1, chunk_n, chunk_size)})
    for i in range(8):
        print np.argmax(y[i])
        plt.subplot(251+i)
        plt.imshow(mnist.test.images[i].reshape(28,28))
    

