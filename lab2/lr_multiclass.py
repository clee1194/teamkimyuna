import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load("notMNIST.npz") as data :
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
hyperparameters['lambda'] = 0.001
hyperparameters['mini_batch_size'] = 500
hyperparameters['num_epochs'] = 100
hyperparameters['eta'] = 0.005

x_size = 28 ** 2
y_size = 10

trainData = np.reshape(trainData, (trainData.shape[0], x_size))
testData = np.reshape(testData, (testData.shape[0], x_size))
validData = np.reshape(validData, (validData.shape[0], x_size))
trainTarget = np.eye(y_size)[trainTarget]
testTarget = np.eye(y_size)[testTarget]
validTarget = np.eye(y_size)[validTarget]

W = tf.Variable(tf.zeros([28**2,10], tf.float32))
b = tf.Variable(tf.zeros([1,10], tf.float32))

x = tf.placeholder(tf.float32, [None, x_size])
x__ = x
y = tf.placeholder(tf.float32, [None, y_size])
learning_rate = tf.placeholder(tf.float32, [])
hp_mbs = tf.placeholder(tf.float32, [])
hp_lambda = tf.placeholder(tf.float32, [])

pred = tf.matmul(x__, W)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) + hp_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b)) / 2

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

adam_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	for i in range(hyperparameters['num_epochs']):
		randIndx = np.arange(3500)
		np.random.shuffle(randIndx)
	
		for j in range(3500/hyperparameters['mini_batch_size']):
			miniData = trainData[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]
			miniTarget = trainTarget[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]

			# Adam Optimizer

			_, c = sess.run([adam_op, loss], feed_dict={x:miniData, y:miniTarget, learning_rate:hyperparameters['eta'], 
				hp_lambda:hyperparameters['lambda']})
		
		validError = sess.run(loss, feed_dict={x:validData, y:validTarget, hp_lambda:hyperparameters['lambda']})
		trainError = sess.run(loss, feed_dict={x:trainData, y:trainTarget, hp_lambda:hyperparameters['lambda']})

		validAccuracy = sess.run(accuracy, feed_dict={x:validData, y:validTarget, hp_lambda:hyperparameters['lambda']})
		trainAccuracy = sess.run(accuracy, feed_dict={x:trainData, y:trainTarget, hp_lambda:hyperparameters['lambda']})

		print "\n--- Epoch %d ---" % (i)
		print "Valid Error: ", validError
		print "Training Error: ", trainError
		print "Valid Accuracy: ", validAccuracy
		print "Training Accuracy: ", trainAccuracy

