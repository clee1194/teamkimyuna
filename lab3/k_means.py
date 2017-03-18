import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import Counter

data = np.load('data2D.npy')


def getdist(X,Y):
	
	XX = tf.reshape(tf.reduce_sum(tf.multiply(X,X),1),[-1,1])
	YY = tf.reshape(tf.reduce_sum(tf.multiply(tf.transpose(Y),tf.transpose(Y)),0),[1,-1])
	XY = tf.scalar_mul(2.0,tf.matmul(X,tf.transpose(Y)))

	return XX - XY + YY


def k_means(K,data,LEARNINGRATE,epochs,valid_start):

	train_data = data[:valid_start]
	valid_data = data[valid_start:]

	N = train_data.shape[0]
	D = train_data.shape[1]

	x = tf.placeholder("float32", [None,D])

	mu = tf.Variable(tf.truncated_normal([K,D],stddev=0.25))

	dist = getdist(x,mu)

	min_d = tf.reduce_min(dist,1)

	loss = tf.reduce_sum(min_d,0)

	adamop = tf.train.AdamOptimizer(LEARNINGRATE, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	assign = tf.argmin(dist,1)

	init = tf.initialize_all_variables()

	train_loss = np.zeros(epochs)


	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(epochs):
			L, _ = sess.run([loss, adamop], feed_dict={x:train_data})
			train_loss[epoch] = L 

		cluster_assign = sess.run(assign, feed_dict={x:train_data})

		valid_loss = sess.run(loss, feed_dict={x:valid_data})

	return train_loss, cluster_assign, valid_loss


epochs = 600
K = 3


### PART 2 ### 

# valid_loss1 = []
# valid_loss2 = []
# valid_loss3 = []


# for eta in [0.1, 0.01, 0.001]:

# 	validError, cluster_assign, _ = k_means(K, data, eta, epochs, len(data))
	
# 	if eta == 0.1:
# 		valid_loss1 = validError
# 	if eta == 0.01:
# 		valid_loss2 = validError
# 	if eta == 0.001:
# 		valid_loss3 = validError

# 	print eta

# plt.figure()
# plt.plot(range(epochs),valid_loss1[:],label="0.1",linewidth=0.75)
# plt.plot(range(epochs),valid_loss2[:],label="0.01",linewidth=0.75)
# plt.plot(range(epochs),valid_loss3[:],label="0.001",linewidth=0.75)
# plt.legend(loc='best')
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()



LEARNINGRATE = 0.01

# train_loss, cluster_assign, _ = k_means(K, data, LEARNINGRATE, epochs, len(data))
# print 'Minimum Training Loss: ', train_loss.min()
# print 'Minimum Validation Loss: ', valid_loss.min()

# plt.figure()
# plt.plot(range(epochs),train_loss)
# plt.title('Loss vs. Number of Epochs')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.show()



### PART 3 ###


# for k in range(1,6):
# 	print k 

# 	train_loss, cluster_assign, _ = k_means(k, data, LEARNINGRATE, epochs, len(data))
# 	print 'Minimum Training Loss: ', train_loss.min()

# 	samples = dict(Counter(cluster_assign))
# 	samples.update((x,y*100.0/data.shape[0]) for x,y in samples.items())
# 	print '% of points in each cluster: ', samples 

# 	# plot
# 	colors = ['c','r','g','m','y']
# 	for i in range(k):
# 		cluster_data = data[:len(cluster_assign)][cluster_assign==i].T
# 		if i == 0:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 1")
# 		if i == 1:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 2")
# 		if i == 2:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 3")
# 		if i == 3:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 4")
# 		if i == 4:
# 			plt.scatter(cluster_data[0],cluster_data[1],color=colors[i],label="K = 5")
# 		plt.legend(loc='best')
# 	plt.show()



### PART 4 ###

# for k in range(1,6):
# 	print k

# 	train_loss, cluster_assign, valid_loss = k_means(k, data, LEARNINGRATE, epochs, 2*len(data)/3)
# 	print 'Minimum Training Loss: ', train_loss.min()
# 	print 'Minimum Validation Loss: ', valid_loss.min()
