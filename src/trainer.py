import tensorflow as tf
import numpy as np
import model
import batch

DATA_DIR = './ORIGINAL_SOUNDS_THREE_LABEL_DATA'
MODEL_DIR = './P2_UNLOADED' #'./P2_UNLOADED' or './P2_LOADED'

print('Loading data from disk...')
dataset = np.loadtxt(DATA_DIR + '/ORIGINAL_SOUNDS_THREE_LABEL_DATA')
print('Load successful')

y_data = dataset[:,0]
x_data = dataset[:,2:-4]

for i in range(len(y_data)):
    y_data[i] = 1 if y_data[i] == -1.0 else 0 # if UNLOADED, 1 if y_data[i] == -1.0 else 0
                                             # if LOADED, 1 if y_data == 1.0 else 0
y_data = np.reshape(y_data, [-1,1])
print(x_data.shape)
print(y_data.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.Session() as sess:
    modelmaker = model.NetworkModel()
    bat = batch.Batch(x_data[0:50000,:], y_data[0:50000,:])
    Train, Hypothesis_prob, Cost, Saver = modelmaker.get_model(sess, MODEL_DIR, X, Y)

    Prediction = tf.floor(Hypothesis_prob+0.5)
    Correct = tf.equal(Prediction, Y)
    Accuracy = tf.reduce_mean(tf.cast(Correct, tf.float32))

    for step in range(15000):
        x_batch, y_batch = bat.next_batch(128)
        #y_batch = np.reshape(y_batch, [-1,1])
        sess.run(Train, feed_dict={X:x_batch, Y:y_batch})

        if step % 100 == 0:
            print(step, sess.run([Cost, Accuracy], feed_dict={X:x_batch, Y:y_batch}))
    Saver.save(sess, MODEL_DIR + '/model.ckpt')
