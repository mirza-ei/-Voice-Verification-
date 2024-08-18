import tensorflow as tf
import numpy as np
import pickle
import librosa
from google.colab import drive
from itertools import combinations

# Mount Google Drive
drive.mount('/content/gdrive')

# Load training and test data
with open('/content/gdrive/My Drive/hw4_trs.pkl', 'rb') as f:
    train_file = pickle.load(f)
with open('/content/gdrive/hw4_tes.pkl', 'rb') as f:
    test_file = pickle.load(f)

# Convert the files to STFT and get the absolute value
def preprocess_data(data):
    return [np.abs(librosa.stft(s, n_fft=1024, hop_length=512)).T for s in data]

train = preprocess_data(train_file)
test = preprocess_data(test_file)

# Sampling and creating the dataset for train and test
def generate_dataset(data):
    speaker_len = len(data) // 10
    pairs_1, pairs_2, y_bool = [], [], []

    for i in range(1, speaker_len + 1):
        pos_set = data[i*10 - 10:i*10]
        neg_set = np.delete(data, range(i*10 - 10, i*10), axis=0)
        
        # Positive pairs
        idx = list(combinations(range(10), 2))
        l1 = [pos_set[i[0]] for i in idx]
        l2 = [pos_set[j[1]] for j in idx]
        pairs_1.extend(l1)
        pairs_2.extend(l2)
        y_bool.extend([1] * len(idx))

        # Negative pairs
        idx = list(zip(np.random.randint(10, size=45), np.random.randint(len(data) - 10, size=45)))
        l3 = [pos_set[i1[0]] for i1 in idx]
        l4 = [neg_set[j1[1]] for j1 in idx]
        pairs_1.extend(l3)
        pairs_2.extend(l4)
        y_bool.extend([0] * len(idx))

    s = np.arange(np.array(pairs_1).shape[0])
    np.random.shuffle(s)
    return np.array(pairs_1)[s], np.array(pairs_2)[s], np.array(y_bool)[s]

left_train, right_train, y_train = generate_dataset(train)
left_test, right_test, y_test = generate_dataset(test)

# Define GRU cell and batch normalization functions
def gru_cell(hidden_units, dropout):
    return tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=tf.contrib.layers.variance_scaling_initializer()),
        output_keep_prob=1 - dropout
    )

def batch_norm_layer(x, train_phase, momentum=0.9, epsilon=0.001):
    return tf.layers.batch_normalization(
        inputs=x,
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        training=train_phase
    )

# Model parameters
hidden_units = 128
dropout = 0.5
learning_rate = 0.001
n_classes = 64
n_epoch = 10
batch_size = 32

# Reset TensorFlow graph
tf.reset_default_graph()

# Define placeholders
left = tf.placeholder(tf.float32, [None, None, 513])
right = tf.placeholder(tf.float32, [None, None, 513])
y = tf.placeholder(tf.float32, [None, 1])
succ_length = tf.placeholder(tf.int32, None)
flag_training = tf.placeholder(tf.bool)

# Define the Siamese model
def siamese_model(x, flag_training):
    with tf.name_scope("model"):
        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            rnn_cell = tf.contrib.rnn.MultiRNNCell([gru_cell(hidden_units, dropout) for _ in range(2)])
            dynamic_rnn, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32, sequence_length=succ_length)
        with tf.variable_scope("dense", reuse=tf.AUTO_REUSE):
            batchnorm_layer = batch_norm_layer(dynamic_rnn, flag_training)
            dense_layer = tf.layers.dense(
                batchnorm_layer, n_classes,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation=tf.nn.tanh
            )
        return tf.layers.flatten(dense_layer)

# Siamese network outputs
left_output = tf.nn.l2_normalize(siamese_model(left, flag_training), 0)
right_output = tf.nn.l2_normalize(siamese_model(right, flag_training), 0)
layer_dot = tf.reduce_sum(tf.multiply(left_output, right_output), axis=1, keepdims=True)
sig_layer = tf.sigmoid(layer_dot)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=sig_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(sig_layer), y), tf.float32))

# Initialize variables
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# Start TensorFlow session
with tf.Session() as sess:
    sess.run(init_global)
    sess.run(init_local)

    def next_batch(left_data, right_data, labels, lengths, start, batch_size):
        end = start + batch_size
        return left_data[start:end], right_data[start:end], labels[start:end], lengths[start:end]

    # Training loop
    for epoch in range(n_epoch):
        for i in range(len(left_train) // batch_size):
            epoch_x1, epoch_x2, epoch_y, seq_len_batch = next_batch(left_train, right_train, y_train, left_train_len, i * batch_size, batch_size)
            epoch_y = epoch_y.reshape([batch_size, 1])
            _, c = sess.run([optimizer, loss], feed_dict={left: epoch_x1, right: epoch_x2, y: epoch_y, succ_length: seq_len_batch, flag_training: True})

        epoch_loss, acc = sess.run([loss, accuracy], feed_dict={left: left_test, right: right_test, y: y_test.reshape([len(y_test), 1]), succ_length: left_test_len, flag_training: False})
        print(f"Epoch {epoch} Test Loss = {epoch_loss} Test Accuracy = {acc}")

        if acc >= 0.70:
            break
