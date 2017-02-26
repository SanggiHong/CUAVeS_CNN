import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt

DATA_DIR = './ORIGINAL_SOUNDS_THREE_LABEL_DATA'

print ('Loading data...')
testset = np.loadtxt(DATA_DIR + '/ORIGINAL_SOUNDS_THREE_LABEL_DATA')
print('Load successful')

x_data = testset[50000:,2:-4] #Use frequency 100Hz~8000Hz
print(x_data.shape)

y_data = testset[50000:,0]
y_data = np.reshape(y_data, [-1,1])
print(y_data.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.Session() as sess:
    modelmaker = model.NetworkModel()
    _, Hypothesis_prob_load, _, saver = modelmaker.get_model(sess, './P2_LOADED', X, Y)
    prob_load = sess.run([Hypothesis_prob_load], feed_dict={X:x_data})

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    modelmaker = model.NetworkModel()

with tf.Session(graph=g) as sess:
    _, Hypothesis_prob_unload, _, _ = modelmaker.get_model(sess, './P2_UNLOADED', X, Y)
    prob_unload = sess.run([Hypothesis_prob_unload], feed_dict={X:x_data})

prob_unload = np.reshape(np.array(prob_unload), -1)
prob_load = np.reshape(np.array(prob_load), -1)

result = []
for i in range(len(x_data)):
    if (prob_load[i] > 0.5):
        result.append([1])
    elif (prob_unload[i] > 0.5):
        result.append([-1])
    else:
        result.append([0])
result = np.array(result)

noise = 0
unloaded_r = 0
unloaded = 0
loaded = 0
loaded_r = 0
for i in range(len(y_data)):
    if (y_data[i] == 0 and result[i] == 0):
        noise = noise+1
    elif (y_data[i] == -1. and result[i] == -1):
        unloaded = unloaded+1
    elif (y_data[i] == 1. and result[i] == 1):
        loaded = loaded+1

    if (y_data[i] == -1):
        unloaded_r = unloaded_r+1
    elif (y_data[i] == 1):
        loaded_r = loaded_r+1
    #print(prob_unload[i], prob_load[i], y_data[i], result[i])
print(unloaded, unloaded_r)
print(loaded, loaded_r)
print(np.mean(np.equal(result, y_data)))
