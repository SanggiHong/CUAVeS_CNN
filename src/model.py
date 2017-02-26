import tensorflow as tf
import os

class NetworkModel:
    def get_model_debug(self, session, model_directory, input_data):
        self.create_model(session, model_directory, input_data)
        self.Hypothesis_prop = tf.nn.sigmoid(self.Output)
        
        self.Saver = tf.train.Saver()
        self.checkpoint = tf.train.get_checkpoint_state(model_directory)
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            print('The model loaded from', self.checkpoint.model_checkpoint_path)
            self.Saver.restore(session, self.checkpoint.model_checkpoint_path)
        else:
            print('The model doesn\'t exist')
            init = tf.global_variables_initializer()
            session.run(init)
        
        return {'Filter':self.Filter,
                'Weight':self.Weight,
                'Bias':self.Bias,
                'Filtered_data':self.Filtered_data,
                'Subsampled_data':self.Subsampled_data,
                'Vectorized_data':self.Vectorized_data,
                'Output':self.Output}, \
                self.Hypothesis_prop, \
                self.Saver
        
    def get_model(self, session, model_directory, input_data, label_data):
        if model_directory and not os.path.isdir(model_directory):
            print(model_directory + ' directory is created')
            os.makedirs(model_directory)
        self.create_model(session, model_directory, input_data)
        self.Cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.Output, label_data))
        self.Train = tf.train.AdamOptimizer(0.1).minimize(self.Cost)
        self.Hypothesis_prop = tf.nn.sigmoid(self.Output)

        self.Saver = tf.train.Saver()
        self.checkpoint = tf.train.get_checkpoint_state(model_directory)
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            print('The model loaded from', self.checkpoint.model_checkpoint_path)
            self.Saver.restore(session, self.checkpoint.model_checkpoint_path)
        else:
            print('The model doesn\'t exist')
            init = tf.global_variables_initializer()
            session.run(init)

        return self.Train, self.Hypothesis_prop, self.Cost, self.Saver

    def create_model(self, session, model_directory, input_data):
        freq_width = 85
        conv_filter_size = 10
        conv_step_size = 1
        conv_filter_count = 3
        subsampling_window_size = 5
        subsampling_step_size = 2
        VECTOR_SIZE = int( 
                        (((freq_width/conv_step_size) - subsampling_window_size) / subsampling_step_size + 1) * conv_filter_count
                      )

        with tf.variable_scope('CNN-FCNN'):  
            self.Reshaped_X = tf.reshape(input_data, shape=[-1,1,freq_width,1])

            with tf.variable_scope('Convolution'):
                self.Filter = tf.get_variable(shape=[1,conv_filter_size,1,conv_filter_count], dtype=tf.float32, name='Filter')
                self.Filtered_data = tf.nn.conv2d(self.Reshaped_X, self.Filter, strides=[1,1,conv_step_size,1], padding='SAME')
                self.Filtered_data_act = tf.nn.relu(self.Filtered_data)
                self.Subsampled_data = tf.nn.max_pool(self.Filtered_data_act, ksize=[1,1,subsampling_window_size,1], strides=[1,1,subsampling_step_size,1], padding='VALID')
            with tf.variable_scope('NeuralNet'):
                self.Weight = tf.get_variable(shape=[VECTOR_SIZE,1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), name='Weight')
                self.Bias = tf.get_variable(shape=[1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), name='Bias')
                self.Vectorized_data = tf.reshape(self.Subsampled_data, shape=[-1,VECTOR_SIZE])
                self.Output = tf.matmul(self.Vectorized_data, self.Weight) + self.Bias
