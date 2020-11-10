#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:24:07 2018

@author: dlz
"""

import tensorflow as tf
import h5py
from spatial_3Dtransformer import transform
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# %% Load data
with h5py.File("./data/3d-mnist/full_dataset_vectors.h5", "r") as hf:    
     X_train = hf["X_train"][:]
     y_train = hf["y_train"][:]    
     X_test = hf["X_test"][:]  
     y_test = hf["y_test"][:]
     
# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

# %% Graph representation of our network

# %% Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None, 4096])
y = tf.placeholder(tf.float32, [None, 10])

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 16, 16, 16, 1])


filter_size = 3
keep_prob = tf.placeholder(tf.float32)

# %% We'll setup the there-layer localisation network to figure out the
# %% parameters for an rotation transformation of the input
f1 = 8
w1 = weight_variable([filter_size, filter_size, filter_size, 1, f1])
b1 = bias_variable([f1])
h1 = tf.nn.relu(
    tf.nn.conv3d(input=x_tensor,
                 filter=w1,
                 strides=[1, 2, 2, 2, 1],
                 padding='SAME') +b1)
f2 = 4
w2 = weight_variable([filter_size, filter_size, filter_size, f1, f2])
b2 = bias_variable([f2])
h2 = tf.nn.relu(
    tf.nn.conv3d(input=h1,
                 filter=w2,
                 strides=[1, 2, 2, 2, 1],
                 padding='SAME') +b2)

h2 = tf.reshape(h2,[-1,256])
w3 = weight_variable([256, 4])
b3 = bias_variable([4])
h_fc_loc2 = tf.nn.tanh(tf.matmul(h2, w3) + b3)


# %% We'll create a spatial transformer module to identify discriminative patches
out_size = 16
h_trans = transform(x_tensor, h_fc_loc2, out_size)

# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x depth x input_channels x output_channels]

# %% Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x depth x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.

# here,if you don't want to have a transform for mnist recognition for a comparison,you can use 
# x_tensor to replace h_trans

n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, filter_size, 1, n_filters_1])

# %% Bias is [output_channels]
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
    tf.nn.conv3d(input=h_trans,
                 filter=W_conv1,
                 strides=[1, 2, 2, 2, 1],
                 padding='SAME') + b_conv1)

# %% And just like the first layer, add additional layers to create
# a deep net
n_filters_2 = 8
W_conv2 = weight_variable([filter_size, filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv3d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 2, 1],
                 padding='SAME') + b_conv2)

# %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 4 * 4 *4 * n_filters_2])

# %% Create a fully-connected layer:
n_fc1 = 512
W_fc1 = weight_variable([4 * 4 * 4* n_filters_2, n_fc1])
b_fc1 = bias_variable([n_fc1])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# %% And finally our softmax layer:
W_fc3 = weight_variable([n_fc1, 10])
b_fc3 = bias_variable([10])
y_logits = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

# %% Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
tf.add_to_collection('losses', cross_entropy)
loss = tf.add_n(tf.get_collection('losses'))

opt = tf.train.AdamOptimizer(learning_rate=0.0001)
optimizer = opt.minimize(loss)

# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
with tf.Session() as sess:
    
    sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
    iter_per_epoch = 100
    n_epochs = 500
    train_size = 10000
    
    indices = np.linspace(0, 10000 - 1, iter_per_epoch)
    indices = indices.astype('int')
    
    batch_size = 100
    magic =  np.random.randint(0,2000,batch_size)
    Loss_1 = np.zeros([n_epochs,iter_per_epoch])
    acc = np.zeros([n_epochs,iter_per_epoch])
    Accuracy = np.zeros([n_epochs])
    for epoch_i in range(n_epochs):
        batch_xt = X_test[magic][:]
        batch_yt = Y_test[magic][:]
        for iter_i in range(iter_per_epoch - 1):
            batch_xs = X_train[indices[iter_i]:indices[iter_i+1]-1][:]
            batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]-1][:]
            if iter_i % 10 == 0:
                [Loss_1[epoch_i,iter_i], acc[epoch_i,iter_i]] = sess.run([cross_entropy, accuracy],feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0})
                print('Iteration: ' + str(iter_i) + ' Loss: ' + str(Loss_1[epoch_i,iter_i]) + ' Accuracy: ' + str(acc[epoch_i,iter_i]))            
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
            Accuracy[epoch_i]=sess.run(accuracy,feed_dict={x: batch_xt,y: batch_yt,keep_prob: 1.0})
        print('Accuracy (%d): ' % epoch_i + str(Accuracy[epoch_i]))
    [theta, x_1, y_1] = sess.run([h_fc_loc2, x_tensor, h_trans],feed_dict={x: X_test, keep_prob: 1.0})
    print(theta[0])

x_2 = x_1[:, :, :, :, 0]
y_2 = y_1[:, :, :, :, 0]
fig = plt.figure()
ax = Axes3D(fig)
m = 0
print(y_test[m])
A = x_2[m]
B = y_2[m]

m = range(0,5)
print(y_test[m])
A = x_2[m]
B = y_2[m]
for v in m:
    fig1 = plt.figure(num=v)
    ax = Axes3D(fig1)
    x = []
    y = []
    z = []
    c = []
    for i in range(16):
        for ii in range(16):
            for iii in range(16):
                if abs(A[v][i][ii][iii])>0.0001:
                    x.append(i)
                    y.append(ii)
                    z.append(iii)
                    c.append(A[v][i][ii][iii])
    ax.scatter(x, y, z, c=c)
    
    
    fig2 = plt.figure(num=v+10)
    ax = Axes3D(fig2)
    xx = []
    yy = []
    zz = []
    cc = []
    for i in range(16):
        for ii in range(16):
            for iii in range(16):
                if abs(B[v][i][ii][iii])>0.0001:
                    xx.append(i)
                    yy.append(ii)
                    zz.append(iii)
                    cc.append(B[v][i][ii][iii])
    ax.scatter(xx, yy, zz, c=cc)
#plt.figure()
#plt.plot(Accuracy)