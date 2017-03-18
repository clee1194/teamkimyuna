import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

from collections import Counter


data = np.load('data2D.npy')

def negLL(X, mu, var, pi_var):

	log_pi_gauss = logPiGaussian(X, mu, var, pi)
	sum_log_pi_gauss = tf.reshape(utils.reduce_logsumexp(log_pi_gauss,1),[-1,1])

	tot_sum = tf.reduce_sum(sum_log_pi_gauss,0)

	return -tot_sum


def MoG(K, data, LEARNINGRATE, epochs, valid_start):

	train_data = data[:valid_start]
	valid_data = data[valid_start:]

	B = train_data.shape[0]
	D = train_data.shape[1]

	X = tf.placeholder("float32",shape=[None,D])
	mu = tf.Variable(tf.truncated_normal([K,D],stddev=0.25))
	var = tf.exp(tf.Variable(tf.truncated_normal([1,D],stddev=0.25)))
	pi_var = tf.exp(utils.logsoftmax(tf.Variable(tf.ones([1,D]))))

	L = negLL(X, mu, var, pi_var)

	adam_op = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	log_pi_gauss = logPiGaussian(X, mu, var, pi_var)
	assign = tf.argmax(log_pi_gauss,1)

	init = tf.initialize_all_variables()

	loss_array = np.zeros(epochs)

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			loss, _ = sess.run([L,adam_op],feed_dict={X:train_data})
			loss_array[epoch] = loss

		clus_assign = sess.run(assign, feed_dict={X:train_data})
		mu_, var_, pi_var_ = sess.run([mu,var,pi_var])
		valid_loss = sess.run(L,feed_dict={X:valid_data})

	return loss_array, valid_loss, clus_assign, mu_, var_, pi_var_

K = 3
LEARNINGRATE = 0.004
epochs = 600
valid_start = len(data)

train_loss, _, clus_assign, mu, var, pi_var, = MoG(K, data, LEARNINGRATE, epochs, valid_start)

plt.figure()
plt.plot(range(epochs),train_loss[:],linewidth=0.75)
# plt.plot(range(epochs),valid_loss2[:],label="0.01",linewidth=0.75)
# plt.plot(range(epochs),valid_loss3[:],label="0.001",linewidth=0.75)
#plt.legend(loc='best')
plt.title('Loss vs. Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.show()
