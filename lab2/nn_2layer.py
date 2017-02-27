import numpy as np
import tensorflow as tf
import math

# Good reference for building neural networks on Tensorflow
# https://www.tensorflow.org/versions/master/tutorials/mnist/pros/

with np.load("notMNIST.npz") as data:
	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	trainData, trainTarget = Data[:15000], Target[:15000]
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	testData, testTarget = Data[16000:], Target[16000:]

hyperparameters = {}
hyperparameters['lambda'] = 0.0003
hyperparameters['mini_batch_size'] = 500
hyperparameters['num_epochs'] = 50
hyperparameters['eta'] = 0.0001
hyperparameters['hidden_units'] = 500

x_size = 28 ** 2
y_size = 10

xavier_var = math.sqrt(3.0 / (x_size + y_size)) # 0.106466

trainData = np.reshape(trainData, (15000, x_size))
testData = np.reshape(testData, (testData.shape[0], x_size))
validData = np.reshape(validData, (1000, x_size))
# print trainTarget[:10]
trainTarget = np.eye(y_size)[trainTarget]
testTarget = np.eye(y_size)[testTarget]
validTarget = np.eye(y_size)[validTarget]
# print trainTarget[:10]

def neural_network(x, h_size):
	input_size = x.get_shape().as_list()
	# print input_size
	W = tf.Variable(tf.random_normal((input_size[1], h_size), stddev=xavier_var, dtype=tf.float64))
	# print "weights ", weights
	b = tf.Variable(tf.zeros((1, h_size), tf.float64))
	# print "biases ", biases
	# print "X ", x.get_shape().as_list()
	z = tf.matmul(x, W) + b
	# print "z ", z
	return z, tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

def main():
	learning_rate = tf.placeholder(tf.float32, [])
	x = tf.placeholder(tf.float64, shape=[None, x_size])
	x__ = x
	y_ = tf.placeholder(tf.float64, shape=[None, y_size])
	# print tf.shape(X)

	z_1, loss_1 = neural_network(x__, hyperparameters['hidden_units'])
	# h_1 = tf.nn.relu(z_1)
	z_2, loss_2 = neural_network(z_1, hyperparameters['hidden_units'])

	z_3, loss_3 = neural_network(tf.nn.relu(z_2), y_size)

	pred = tf.nn.relu(z_3)	
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_)) + hyperparameters['lambda'] * (loss_1 + loss_2 + loss_3)
	
	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
	
	adam_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		for epoch in range(hyperparameters['num_epochs']):	
			randIndx = np.arange(15000)
			np.random.shuffle(randIndx)
			for batch in range(15000/hyperparameters['mini_batch_size']):
				miniData = trainData[randIndx[batch*hyperparameters['mini_batch_size']:(batch+1)*hyperparameters['mini_batch_size']]]
				miniTarget = trainTarget[randIndx[batch*hyperparameters['mini_batch_size']:(batch+1)*hyperparameters['mini_batch_size']]]

				# Adam Optimizer

				_, c = sess.run([adam_op, loss], feed_dict={x:miniData, y_:miniTarget, learning_rate:hyperparameters['eta']})
				# print "Epoch %d, Batch %d: Loss = %f" % (epoch, batch, c)

		# print sess.run(tf.reduce_mean(y))
		# print sess.run(tf.one_hot(testTarget[10:20], y_size))
			validError = sess.run(loss, feed_dict={x:validData, y_:validTarget})
			testError = sess.run(loss, feed_dict={x:testData, y_:testTarget})
			validAcc = sess.run(accuracy, feed_dict={x:validData, y_:validTarget})
			testAcc = sess.run(accuracy, feed_dict={x:testData, y_:testTarget})
			print "Epoch %d: Valid Loss = %f, Accuracy = %f" % (epoch, validError, validAcc)
			print "Epoch %d: Test Loss = %f, Accuracy = %f" % (epoch, testError, testAcc)
		

	# print "X size: ", x_size, " Y size: ", y_size
	# print trainTarget[:10]



if __name__ == '__main__':
	main()
