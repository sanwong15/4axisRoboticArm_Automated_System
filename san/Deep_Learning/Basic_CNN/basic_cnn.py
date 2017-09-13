'''
Mar 30, 2017
Author: Hong San Wong
Email: hswong1@uci.edu
This file describe a basic CNN network which perform classification
Structure as follow

INPUT -> CONV1 -> MaxPool -> CONV2 -> MaxPool -> FullyConnected1
 -> FullyConnected2 -> FullyConnected3 -> OUTPUT

 The Follow code should perform trainning and save network parameters

'''


from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

'''
The followings are the self-defined functions and object
'''

# //////////////////////////////  UTILITIES //////////////////////

# Create data object =============================================
class data_object:

    # Constructor
    def __init__(self,img,label):
        self.img = img
        self.label = label # 1 for Hand and 0 for Others

# End of data_object class =======================================

# ////////////////////////////////////////////////////////////////





# /////////////////////////// DATA ////////////////////////////////

'''
Function call flow:
Import data => data_object_arr => shuffer => convertion => next_batch
'''



# Import data
    #from tensorflow.examples.tutorials.mnist import input_data
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
testImg_arr = np.load('testImg_arr.npy')
trainImg_arr = np.load('trainImg_arr.npy')

# Shuffle loaded data arr =======================================
#np.random.shuffle(testImg_arr)
#np.random.shuffle(trainImg_arr)

def shuffle_data(data_array):
    np.random.shuffle(data_array)
    return data_array

# ===============================================================

#  Data Convertion ==============================================
def convert_data(data_array):
    arr_shape = data_array.shape
    arr_W = arr_shape[0] # which in our case, we are expecting 1554
    img_arr_W = data_array[0].img.shape[0] # which in our case, we are epxecting 10000
    # Define img matrix
    img_matrix = np.zeros((arr_W,img_arr_W),dtype=np.int)
    # Define label matrix
    label_matrix = np.zeros((arr_W,n_classes),dtype=np.int)
    # put img_arr into the matrix
    for index in range(arr_W):
        img_matrix[index,:] = data_array[index].img.transpose()
        curr_label = data_array[index].label
        if curr_label == 1:
            label_matrix[index,1] = 1
        else:
            label_matrix[index,0] = 1

    return img_matrix, label_matrix
# ================================================================

# Extract Next Bactch ============================================
def next_batch(data_array,batch_size,start_index):
    temp_arr=[]
    end_index = start_index+batch_size
    temp_arr = data_array[start_index:end_index]
    return temp_arr
# End of Next Batch ==============================================


# ////////////////////////////////////////////////////////////////










# Parameters
learning_rate = 0.001
# Training iteration suppose to be the same as the Training Array Length 
# as we use all the data for training
training_iters = len(trainImg_arr) 
    #batch_size = 128
batch_size = 50    
display_step = 1 # Use this as stopping condition

# Network Parameters
    # n_input = 784 # MNIST data input (img shape: 28*28)
    # n_input is the array size/memory need for each train/test image.
n_input = 10000 # Data input (img shape: 100*100 = 10000)
    # n_classes = 10 # MNIST total classes (0-9 digits)
n_classes = 2 # either is HAND
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input ===================PLACEHOLDER================================
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Add placeholder to identify TRAINING PHASE
# phase_train = tf.placeholder(tf.bool, name = 'phase_train')
# =============================================================================

# Define a conv layer
# Weight is the value within the Kernal
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # Test ZERO PADDING
    # x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID') padding='SAME' => [2,4] i.e: 4 padding on X and Y dimension
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    
    x = tf.nn.bias_add(x, b)

    # Update: April 11, 17
    # Try to add batch nor. so we don't perfer RELU here "tf.nn.relu(x)"
    return x

# Define a maxpool layer
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model (This is a model that take ONE training sample at a time. BUT MAYBE we can psss a batch throught it)
# add a bool term to identify training phase
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Resize our image to 100*100
    # tf.reshape(tensor, shape, name=None)
    x = tf.reshape(x, shape = [-1,100,100,1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Apply Relu Here
    conv1_out = tf.nn.relu(conv1)
    # Max Pooling (down-sampling)
    pool1_out = maxpool2d(conv1_out, k=2)

    # Convolution Layer
    conv2 = conv2d(pool1_out, weights['wc2'], biases['bc2'])
    # Apply Max pool
    conv2_out = tf.nn.relu(conv2)
    # Max Pooling (down-sampling)
    pool2_out = maxpool2d(conv2_out, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2_out, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
    '''
    tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    [5,5,1,32] gives 800 elements
    fully connected, 7*7*64 inputs, 1024 outputs (for original image, it was 28*28 -> after maxpool, it became 14*14 and another maxpool. Result in 7*7)
    with input image: 100*100. Result image: 25*25
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    '''
weights = {

    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([25*25*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model and Performance Evaluate =========================================
# pred = conv_net(x, weights, biases, keep_prob)
# UPDATED to includ phase

# ///////////////////////////// LOSS & Optimizer (TRAINING) ////////////////////////////////
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




'''
The Following Code Initialize the variables + Start the Graph + Start the TRAINING section
'''



# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0

    # We shuffle -> convert data before get into the training loop
    shuffle_data(trainImg_arr)
    train_img_matrix, train_label_matrix = convert_data(trainImg_arr)



    # TRAINING: Keep training until reach max iterations =======================================
    while step * batch_size < training_iters:
        # minst data set has next_batch method but our data set doesn't. So we will have to 
        # batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Initialize trainImg_next_batch
        start_index = step*batch_size
        # We may not need to pre-define an empty vector. Comment it out for now
        #trainImg_next_batch = []
        #batch_x = []
        #batch_y = []

        '''
        # Load data (batch) into trainImg_next_batch
        trainImg_next_batch = next_batch(trainImg_arr,batch_size,start_index)
        print(trainImg_next_batch.shape)

        # Fill up batch_x and batch_y
        for index in range(len(trainImg_next_batch)):
            curr_data_object = trainImg_next_batch[index]
            batch_x.append(curr_data_object.img)
            batch_y.append(curr_data_object.label)


        #batch_x = trainImg_next_batch.img
        #batch_y = trainImg_next_batch.labels
        '''

        batch_x = next_batch(train_img_matrix,batch_size,start_index)
        batch_y = next_batch(train_label_matrix,batch_size,start_index)


        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # END OF TRAINING ======================================================================



    # TESTING: Calculate accuracy for test images ==========================================

    # Define Test img arr and Test label arr
    '''
    testTime_x = []
    testTime_y = []
    for index in range(len(trainImg_next_batch)):
        curr_data_object = trainImg_next_batch[index]
        testTime_x.append(curr_data_object.img)
        testTime_y.append(curr_data_object.label)
    '''
    
    shuffle_data(testImg_arr)
    test_img_matrix, test_label_matrix = convert_data(testImg_arr)


    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_img_matrix,
                                      y: test_label_matrix,
                                      keep_prob: 1.}))

    # END OF TESTING =======================================================================
