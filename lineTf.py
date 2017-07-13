#coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

x = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y_ = tf.placeholder(tf.float32)

y = W*x+b

lost = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(lost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

steps = 10000
xs = np.arange(0, 5, 0.1)
ys = xs*2+ np.random.randint(-2,2,50)
#xs = [1,2,3,4,5,6,7,8,9]
#ys = [4,2,2,3,12,14,10,19,16]
plt.scatter(xs,ys,s=5)
print xs
print ys
for i in range(steps):
    #xs = [i]
#    ys = [3*i]
    feed = {x:xs,y_:ys}
    sess.run(train_step,feed_dict=feed)
    if i%100==0:
        y = sess.run(W)*xs+sess.run(b)
        plt.plot(xs,y)
        plt.pause(0.01)
        print("After %d iteration:" % i)


        print("W: %f" % sess.run(W))
        print("b: %f" % sess.run(b))
        print("lost: %f" % sess.run(lost, feed_dict=feed))
