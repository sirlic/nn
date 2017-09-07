
# coding: utf-8

# In[1]:

import tensorflow as tf


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/', 'Directory for storing data') # 把数据放在/tmp/data文件夹中

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集

print 'data ok-----'
# In[ ]:

x = tf.placeholder(tf.float32, [None, 784]) # 占位符
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)


# In[ ]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(a)+(1-y)*tf.log(1-a),reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

# Test trained model
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# merged_summary_op = tf.merge_all_summaries()
# summary_writer = tf.summary.su('./mnist_logs', sess.graph)
# total_step = 0


for i in range(1000):
    # total_step += 1
    batch_xs,batch_xy = mnist.train.next_batch(10000)
    train.run({x:batch_xs,y:batch_xy})
    # print a.eval()
    print sess.run(cross_entropy,{x:batch_xs,y:batch_xy})
    
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

