import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import draft
import os

DATA_DIR = './DATA_FOR_VISUALIZE'
IMG_DIR = './VISUALIZED_IMAGE'

print ('Loding data...')
dataset = np.loadtxt(DATA_DIR + '/LOADED_UNLOADED_NOISE')
print('Load successful')

if IMG_DIR and not os.path.isdir(IMG_DIR):
    print(IMG_DIR + ' directory is created')
    os.makedirs(IMG_DIR)

x_data = dataset[:,2:-4]
print(x_data.shape)

y_data = dataset[:,0]
y_data = np.reshape(y_data, [-1,1])
print(y_data.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.Session() as sess:
    modelmaker = draft.NetworkModel()
    A, _, _ = modelmaker.get_model_debug(sess, './P2_LOADED', X)

    Filter = sess.run(A['Filter'])
    Weight = sess.run(A['Weight'])
    Bias = sess.run(A['Bias'])
    Filtered_data = sess.run(A['Filtered_data'], feed_dict={X:x_data})
    Subsampled_data = sess.run(A['Subsampled_data'], feed_dict={X:x_data})
    Vectorized_data = sess.run(A['Vectorized_data'], feed_dict={X:x_data})
    Output = sess.run(A['Output'], feed_dict={X:x_data})

    #make graph for filtered data
    print(Filtered_data.shape)
    Feature1 = np.reshape(Filtered_data[0,:,:,0], [1,85])
    Feature2 = np.reshape(Filtered_data[0,:,:,1], [1,85])
    Feature3 = np.reshape(Filtered_data[0,:,:,2], [1,85])
    
    fig, (p1, p2, p3) = plt.subplots(nrows=3)
    p1.imshow(Feature1, cmap='Greys', interpolation='nearest')
    p1.grid(False)
    p2.imshow(Feature2, cmap='Greys', interpolation='nearest')
    p2.grid(False)
    p3.imshow(Feature3, cmap='Greys', interpolation='nearest')
    p3.grid(False)
    fig.savefig(IMG_DIR + '/loadedmodel_loaded_filtereddata.png')
 
    print(Pulled_data.shape)
    Feature1 = np.reshape(Subsampled_data[0,:,:,0], [1,41])
    Feature2 = np.reshape(Subsampled_data[0,:,:,1], [1,41])
    Feature3 = np.reshape(Subsampled_data[0,:,:,2], [1,41])
    
    fig, (p1, p2, p3) = plt.subplots(nrows=3)
    p1.imshow(Feature1, cmap='Greys', interpolation='nearest')
    p1.grid(False)
    p2.imshow(Feature2, cmap='Greys', interpolation='nearest')
    p2.grid(False)
    p3.imshow(Feature3, cmap='Greys', interpolation='nearest')
    p3.grid(False)
    fig.savefig(IMG_DIR + '/loadedmodel_loaded_subsampleddata.png')


"""
    #this part is for visualize filters of model

    print(Filter.shape)
    Feature1 = np.reshape(Filter[:,:,:,0], [1,10])
    Feature2 = np.reshape(Filter[:,:,:,1], [1,10])
    Feature3 = np.reshape(Filter[:,:,:,2], [1,10])

    fig, (p1, p2, p3) = plt.subplots(nrows=3)
    p1.imshow(Feature1, cmap='Greys', interpolation='nearest')
    p1.grid(False)
    p2.imshow(Feature2, cmap='Greys', interpolation='nearest')
    p2.grid(False)
    p3.imshow(Feature3, cmap='Greys', interpolation='nearest')
    p3.grid(False)
    fig.savefig('loadedmodel_filter.png')
"""
