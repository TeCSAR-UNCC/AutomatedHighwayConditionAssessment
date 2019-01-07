
# coding: utf-8

# In[3]:


from __future__ import print_function
import alexnet
import tensorflow as tf
import pandas as pd
import scipy as sp
import sklearn as sk
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np


# Convert labels to np array
all_labels={'RoadwayShoulder':0,'Drainage':1,'Traffic':2,'Roadside':3}
tr_label=[]
va_label=[]
te_label=[]
dataa_labels=[]
def label_conv(data_labels,path):
    
    sdd = pd.read_csv(path,header=None)

    for idx,row in sdd.iterrows():
        if row[0] in all_labels:
                saa = all_labels[row[0]]
                data_labels.append(saa)
    data_labels=np.array(data_labels)
    return data_labels

train_labels = label_conv(tr_label,'train_labels.csv')
valid_labels = label_conv(va_label,'valid_labels.csv')
test_labels = label_conv(te_label,'test_labels.csv')



# Convert images to np array

with tf.Session() as session:
    tr_dataset=[]
    va_dataset=[]
    te_dataset=[]
    g=[]
    def image_conv(conv_dataset,pathee,f,g):
        dir_path = os.path.dirname(os.path.realpath(pathee))
        for i in range(f,g):
            for image_file_name in os.listdir(pathee):
                i=str(i)
                if image_file_name==(i+'.jpg'):
                    filename = dir_path + image_file_name

                    imagee = mpimg.imread(pathee+i+'.jpg')
                    conv_dataset.append(imagee)
        conve_datasett=np.array(conv_dataset)
        return conve_datasett

train_dataset=image_conv(tr_dataset,'./Train/',1,411).astype(float)
valid_dataset=image_conv(va_dataset,'./Validation/',411,461).astype(float)
test_dataset=image_conv(te_dataset,'./Test/',461,544).astype(float)



#Reshaping the datasets and labels
image_size= 448
num_labels = 4
num_channels = 3 # RGB

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)





def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
beta = 0.01

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    def model(x, dropout):
        y=alexnet.classifier(x, dropout)
        return y    

    # Training computation.
    logits = model(tf_train_dataset, dropout=0.7)
    soft_ent=tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    loss = tf.reduce_mean(soft_ent)
    
    with tf.name_scope('l2_loss'):
        l2_loss = tf.reduce_sum(beta * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
        tf.summary.scalar('l2_loss', l2_loss)

    loss = tf.reduce_mean(loss + l2_loss)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, dropout=1.0))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, dropout=1.0))


num_steps= 10000

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        logend=session.run(
          [logits], feed_dict=feed_dict)
        soflos=session.run(
          [soft_ent], feed_dict=feed_dict) 
        
        if (step % 50 == 0):
            print('softmax_cross_entropy is', soflos)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
        
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

