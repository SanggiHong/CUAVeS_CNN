import tensorflow as tf
import numpy as np
import model
import matplotlib.pyplot as plt
import sys
import time

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

print('Now classifying...')
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

loaded_hit = 0
unloaded_hit = 0
noise_hit = 0
loaded_miss = 0
unloaded_miss = 0
noise_miss = 0
sys.stdout.write('┌───────────┬─────────────┬──────────┬────────────┬──────────────┬───────────┐\n')
sys.stdout.write('│Loaded(hit)│Unloaded(hit)│Noise(hit)│Loaded(miss)│Unloaded(miss)│Noise(miss)│\n')
sys.stdout.write('├───────────┼─────────────┼──────────┼────────────┼──────────────┼───────────┤\n')
for i in range(len(x_data)):
    if (prob_load[i] > 0.5):
        result.append([1])
        if (y_data[i] == 1):
            loaded_hit = loaded_hit+1
        else: 
            loaded_miss = loaded_miss+1
    elif (prob_unload[i] > 0.5):
        result.append([-1])
        if (y_data[i] == -1):
            unloaded_hit = unloaded_hit+1
        else:
            unloaded_miss = unloaded_miss+1
    else:
        result.append([0])
        if (y_data[i] == 0):
            noise_hit = noise_hit+1
        else:
            noise_miss = noise_miss+1
    sys.stdout.write('│%11s│%13s│%10s│%12s│%14s│%11s│\r' %(loaded_hit, unloaded_hit, noise_hit, loaded_miss, unloaded_miss, noise_miss))
    time.sleep(0.0005)

sys.stdout.write('│%11s│%13s│%10s│%12s│%14s│%11s│\n' %(loaded_hit, unloaded_hit, noise_hit, loaded_miss, unloaded_miss, noise_miss))
sys.stdout.write('└───────────┴─────────────┴──────────┴────────────┴──────────────┴───────────┘\n')
result = np.array(result)

print('Accuracy :', np.mean(np.equal(result, y_data)))
